import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from timm.models.layers import DropPath, trunc_normal_


def load_era5_data(step):
    """Acá debo cargar un paso del era5, pero por ahora simplemente carguemos un dummy."""

    # AIRE: 5 CANALES en 13 NIVELES DE PRESION, 721x1440
    input_data = torch.randn(1, 5, 13, 721, 1440)
    # SUPERFICIE: 4 CANALES, 721x1440
    input_surface_data = torch.randn(1, 4, 721, 1440)
    
    # Target sería t + delta_t
    target_data = torch.randn(1, 5, 13, 721, 1440)
    target_surface_data = torch.randn(1, 4, 721, 1440)
    
    return input_data, input_surface_data, target_data, target_surface_data

def load_constant_masks():
    """Lo mismo que cargar el era5, hago un dummy"""
    # Cargamos un canal por mascara 
    land_mask = torch.randn(1, 1, 721, 1440)
    soil_type = torch.randn(1, 1, 721, 1440)
    topography = torch.randn(1, 1, 721, 1440)
    return land_mask, soil_type, topography

def load_static_data():
    """dummy"""
    # 5 variables de aire y 4 de suelo
    weather_mean = torch.randn(1, 5, 1, 1, 1)
    weather_std = torch.randn(1, 5, 1, 1, 1)
    weather_surface_mean = torch.randn(1, 4, 1, 1)
    weather_surface_std = torch.randn(1, 4, 1, 1)
    return weather_mean, weather_std, weather_surface_mean, weather_surface_std
    
def get_model_path(lead_time):
    """Funci{ion que carga los paths d elos modelos}."""
    return f"pangu_weather_{lead_time}h.pt"


class MLP(nn.Module):
    """Multi Layer Perceptron"""
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, in_features * 4)
        self.act = act_layer()
        self.fc2 = nn.Linear(in_features * 4, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_chans_surface, in_chans_upper, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans_surface = in_chans_surface
        self.in_chans_upper = in_chans_upper
        self.embed_dim = embed_dim
        
        # Se utiliza convolucion para particionar la data en cubos
        self.proj_upper = nn.Conv3d(in_chans_upper, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Agregamos las mascaras
        self.proj_surface = nn.Conv2d(in_chans_surface + 3, embed_dim, kernel_size=patch_size[1:], stride=patch_size[1:])

        # Cargamos las mascaras y las dejamos en un buffer (esto asegura que despues se muevan al device correcto)
        land_mask, soil_type, topography = load_constant_masks()
        self.register_buffer("land_mask", land_mask)
        self.register_buffer("soil_type", soil_type)
        self.register_buffer("topography", topography)

    def forward(self, x_upper, x_surface):
        B, C_upper, P, H, W = x_upper.shape
        B, C_surface, _, _ = x_surface.shape
        
        # Pad input para que pueda ser divisible por path size
        # Niveles de presion (13 -> 14), H: (721 -> 724), W: (1440 -> 1440)
        pad_p = (self.patch_size[0] - P % self.patch_size[0]) % self.patch_size[0]
        pad_h = (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1]
        pad_w = (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2]
        
        x_upper = F.pad(x_upper, (0, pad_w, 0, pad_h, 0, pad_p))
        x_surface = F.pad(x_surface, (0, pad_w, 0, pad_h))

        # Tengo que llevarlas con el mismo padding a al mismo tamaño que las de superficie
        land_mask_padded = F.pad(self.land_mask, (0, pad_w, 0, pad_h))
        soil_type_padded = F.pad(self.soil_type, (0, pad_w, 0, pad_h))
        topography_padded = F.pad(self.topography, (0, pad_w, 0, pad_h))

        # Proyectamos las variables de aire y superficie en los parches
        x_upper = self.proj_upper(x_upper)  # [B, D, P_patch, H_patch, W_patch]
        
        # Concatenamos los campos que son constantes a variables de superficie y expandimos las mascaras para que coincidan con el batch
        b_size = x_surface.size(0)
        x_surface = torch.cat((
            x_surface,
            land_mask_padded.expand(b_size, -1, -1, -1),
            soil_type_padded.expand(b_size, -1, -1, -1),
            topography_padded.expand(b_size, -1, -1, -1)
        ), dim=1)
        x_surface = self.proj_surface(x_surface) # [B, D, H_patch, W_patch]

        # Agregamos una dimension a los parches de superficie para concatenar con los de aire
        x_surface = x_surface.unsqueeze(2) # [B, D, 1, H_patch, W_patch]
        
        # Concatenar en niveles de presion
        x = torch.cat((x_surface, x_upper), dim=2) # [B, D, P_patch_total, H_patch, W_patch]
        
        # Reshape para los bloques de transformers
        # [B, D, Z, H, W] -> [B, Z, H, W, D] -> [B, Z*H*W, D]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(B, -1, self.embed_dim)
        return x

class PatchRecovery(nn.Module):
    def __init__(self, patch_size, out_chans_surface, out_chans_upper, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.out_chans_surface = out_chans_surface
        self.out_chans_upper = out_chans_upper
        self.embed_dim = embed_dim
        
        # Usan dos convoluciones transpuestas para recuperar data
        self.proj_upper = nn.ConvTranspose3d(embed_dim, out_chans_upper, kernel_size=patch_size, stride=patch_size)
        self.proj_surface = nn.ConvTranspose2d(embed_dim, out_chans_surface, kernel_size=patch_size[1:], stride=patch_size[1:])

    def forward(self, x, Z, H, W):
        B, _, D = x.shape
        
        # Reshape x back to a 3D grid of patches
        # [B, Z*H*W, D] -> [B, Z, H, W, D] -> [B, D, Z, H, W]
        x = x.view(B, Z, H, W, D)
        x = x.permute(0, 4, 1, 2, 3).contiguous() # [B, D, Z, H, W]
        
        # SSeparamos aire y superficie
        x_surface = x[:, :, 0, :, :] # [B, D, H, W]
        x_upper = x[:, :, 1:, :, :] # [B, D, Z-1, H, W]
        
        # Aplicar convolucioness
        output_upper = self.proj_upper(x_upper)
        output_surface = self.proj_surface(x_surface)
        
        # Crop para sacar los paddings en cero (13, 721, 1440)
        output_upper = output_upper[:, :, :13, :721, :1440]
        output_surface = output_surface[:, :, :721, :1440]

        return output_upper, output_surface

class DownSample(nn.Module):
    """Down-sampling operation."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_dim)
        self.linear = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x, Z, H, W):
        B, _, C = x.shape
        x = x.view(B, Z, H, W, C)
        
        # Pad
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        _, _, H_pad, W_pad, _ = x.shape
        
        # Space ->> depth
        x = x.view(B, Z, H_pad // 2, 2, W_pad // 2, 2, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = x.view(B, -1, 4 * C) # Shape: [B, Z*H/2*W/2, 4*C]
        
        x = self.norm(x)
        x = self.linear(x)
        
        return x

class UpSample(nn.Module):
    """UpSample de toda la vida"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, 4 * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        
    def forward(self, x, Z, H, W):
        B, _, C = x.shape
        x = self.linear1(x) # [B, Z*H*W, 4*out_dim]
        
        C_out = C // 2 
        # Depth ->> space
        x = x.view(B, Z, H, W, 2, 2, C_out)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = x.view(B, -1, C_out) # [B, Z*2H*2W, out_dim]

        # Falta el crop que hacen al input shape de la red
        
        x = self.norm(x)
        x = self.linear2(x)
        return x

class EarthSpecificBlock(nn.Module):
    """
    3D transformer block with Earth-Specific bias and window attention.
    """
    def __init__(self, dim, num_heads, window_size=(2, 6, 12), mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
    
        self.attn = EarthAttention3D(dim, num_heads=num_heads, window_size=self.window_size)
    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, Z, H, W, roll):
        shortcut = x
        x = self.norm1(x)
        
        B, L, C = x.shape
        x = x.view(B, Z, H, W, C)

        # Pad feature maps to be divisible by window size
        pad_z = (self.window_size[0] - Z % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x_pad = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_z))
        _, Z_pad, H_pad, W_pad, _ = x_pad.shape
        
        attn_mask = None
        # Cyclic shift y creación de la máscara para shifted window attention
        if roll:
            shift_z, shift_h, shift_w = self.window_size[0] // 2, self.window_size[1] // 2, self.window_size[2] // 2
            x_pad = torch.roll(x_pad, shifts=(-shift_z, -shift_h, -shift_w), dims=(1, 2, 3))

            img_mask = torch.zeros((1, Z_pad, H_pad, W_pad, 1), device=x.device)
            slices = (slice(0, -self.window_size[0]),
                      slice(-self.window_size[0], -shift_z),
                      slice(-shift_z, None))
            cnt = 0
            for z in slices:
                for h in slices:
                    for w in slices:
                        img_mask[:, z, h, w, :] = cnt
                        cnt += 1
            
            mask_windows = self._window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        # Partition into windows
        windows = self._window_partition(x_pad, self.window_size)
        windows = windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        
        # Aplicar window attention con la máscara (si existe)
        attn_windows = self.attn(windows, mask=attn_mask)

        # Merge ventanas
        x_shifted = self._window_reverse(attn_windows, self.window_size, Z_pad, H_pad, W_pad)

        # Reverse cyclic shift
        if roll:
            x_shifted = torch.roll(x_shifted, shifts=(shift_z, shift_h, shift_w), dims=(1, 2, 3))

        # Remove padding
        x = x_shifted[:, :Z, :H, :W, :].contiguous()
        x = x.view(B, L, C)

        # Capas finales del bloque
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def _window_partition(self, x, window_size):
        B, Z, H, W, C = x.shape
        wz, wh, ww = window_size
        x = x.view(B, Z // wz, wz, H // wh, wh, W // ww, ww, C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return windows

    def _window_reverse(self, windows, window_size, Z, H, W):
        wz, wh, ww = window_size
        B = int(windows.shape[0] / (Z * H * W / wz / wh / ww))
        x = windows.view(B, Z // wz, H // wh, W // ww, wz, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Z, H, W, -1)
        return x

class EarthSpecificLayer(nn.Module):
    """Capa del Earth Specific transformer"""
    def __init__(self, depth, dim, num_heads, drop_path_list):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList([
            EarthSpecificBlock(dim=dim, num_heads=num_heads, drop_path=drop_path_list[i])
            for i in range(depth)
        ])
        
    def forward(self, x, Z, H, W):
        for i, blk in enumerate(self.blocks):
            # Logica de shifted windows haciendo roll cada 2 bloques
            roll = (i % 2 != 0)
            x = blk(x, Z, H, W, roll=roll)
        return x

class EarthAttention3D(nn.Module):
    '''
    3D window attention with the Earth-Specific bias, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    '''
    def __init__(self, dim, num_heads, window_size=(2, 6, 12), dropout_rate=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Capa lineal para proyectar la entrada a Query, Key y Value
        self.qkv = nn.Linear(dim, dim * 3, bias=True) # Linear1
        self.proj = nn.Linear(dim, dim) # Linear2
        self.proj_drop = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

        # Implementación del Earth-Specific Bias
        # Crear el índice de posición relativa para una ventana 3D.
        self._construct_index()

        # Tabla de parámetros de sesgo aprendibles.
        # El tamaño es el número de posibles posiciones relativas x el número de cabezas.
        num_relative_positions = (2 * self.window_size[0] - 1) * \
                                 (2 * self.window_size[1] - 1) * \
                                 (2 * self.window_size[2] - 1)
        
        self.earth_specific_bias = nn.Parameter(torch.zeros(num_relative_positions, num_heads))
        trunc_normal_(self.earth_specific_bias, std=.02)

    def _construct_index(self):
        """Construye el índice para reutilizar los parámetros de sesgo simetricos."""
        wz, wh, ww = self.window_size
        
        # Crea coordenadas para cada eje dentro de la ventana
        coords_z = torch.arange(wz)
        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)

        # Crea una malla de coordenadas 3D
        coords = torch.stack(torch.meshgrid([coords_z, coords_h, coords_w], indexing="ij")) # Forma: [3, Wz, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1) # [3, Wz*Wh*Ww]

        # Calcula la diferencia relativa entre cada par de puntos
        # [3, N, 1] - [3, 1, N] -> [3, N, N] donde N = Wz*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # Forma: [N, N, 3

        # Desplaza las coordenadas para que sean índices positivos
        relative_coords[:, :, 0] += wz - 1
        relative_coords[:, :, 1] += wh - 1
        relative_coords[:, :, 2] += ww - 1

        # Convierte las coordenadas 3D relativas a un índice 1D plano
        relative_coords[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        relative_coords[:, :, 1] *= (2 * ww - 1)
        
        relative_position_index = relative_coords.sum(-1) # [N, N]
        
        # Guarda el índice como un buffer (no es un parámetro entrenable)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        B_, N, C = x.shape # B_ es Batch*num_windows, N es window_size**3, C es dim

        # Proyección QKV
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Escalar el query
        q = q * self.scale

        # Calcular puntuaciones de atención
        attn = (q @ k.transpose(-2, -1))

        # Añadir el sesgo posicional específico de la Tierra
        bias = self.earth_specific_bias[self.relative_position_index.view(-1)].view(N, N, -1)
        bias = bias.permute(2, 0, 1).contiguous().unsqueeze(0) # [1, num_heads, N, N]
        attn = attn + bias
        
        # Aplicar la máscara si existe (para ventanas desplazadas)
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        attn = self.proj_drop(attn)

        # Mezclar con los valores (V)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        # Proyección final
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PanguModel(nn.Module):
    def __init__(self, patch_size=(2, 4, 4), embed_dim=192, depths=(2, 6, 6, 2), num_heads=(6, 12, 12, 6)):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self._input_layer = PatchEmbedding(patch_size, in_chans_surface=4, in_chans_upper=5, embed_dim=embed_dim)

        # Profundidad estocastica
        dpr = [x.item() for x in torch.linspace(0, 0.2, sum(depths))]
        
        # Four basic layers (Encoder & Decoder)
        self.layer1 = EarthSpecificLayer(depth=depths[0], dim=embed_dim, num_heads=num_heads[0], drop_path_list=dpr[:depths[0]])
        self.layer2 = EarthSpecificLayer(depth=depths[1], dim=embed_dim * 2, num_heads=num_heads[1], drop_path_list=dpr[depths[0]:sum(depths[:2])])
        self.layer3 = EarthSpecificLayer(depth=depths[2], dim=embed_dim * 2, num_heads=num_heads[2], drop_path_list=dpr[sum(depths[:2]):sum(depths[:3])])
        self.layer4 = EarthSpecificLayer(depth=depths[3], dim=embed_dim, num_heads=num_heads[3], drop_path_list=dpr[sum(depths[:3]):])
        
        # Downsample and Upsample layers
        self.downsample = DownSample(in_dim=embed_dim, out_dim=embed_dim * 2)
        self.upsample = UpSample(in_dim=embed_dim * 2, out_dim=embed_dim)

        # Patch Recovery
        self._output_layer = PatchRecovery(patch_size, out_chans_surface=4, out_chans_upper=5, embed_dim=embed_dim * 2)

    def forward(self, input_upper, input_surface):
        """Backbone architecture"""
        # Embed the input fields into patches
        x = self._input_layer(input_upper, input_surface)
        
        # Guardar las dimensiones originales de los parches
        Z, H_orig, W_orig = 8, 181, 360
        
        # ---- Encoder ----
        # Layer 1
        x = self.layer1(x, Z, H_orig, W_orig)
        skip = x
        
        # Downsample
        x = self.downsample(x, Z, H_orig, W_orig)
        H_down = (H_orig + 1) // 2
        W_down = W_orig // 2
        
        # Layer 2
        x = self.layer2(x, Z, H_down, W_down)
        
        # ---- Decoder ----
        # Layer 3
        x = self.layer3(x, Z, H_down, W_down)
        
        # Upsample
        x = self.upsample(x, Z, H_down, W_down)
        
        # Recortar el padding añadido durante el downsampling
        B, _, C = x.shape
        H_up, W_up = H_down * 2, W_down * 2
        
        x = x.view(B, Z, H_up, W_up, C)
        x = x[:, :, :H_orig, :W_orig, :].contiguous() # Recortamos a las dimensiones originales

        
        # Layer 4
        # Ahora H y W deben ser las originales para esta capa
        x = self.layer4(x, Z, H_orig, W_orig)
        
        # Skip connection
        x = torch.cat((skip, x), dim=-1) # Ahora las dimensiones coinciden
        
        # Recover the output fields from patches
        output_upper, output_surface = self._output_layer(x, Z, H_orig, W_orig)
        return output_upper, output_surface

