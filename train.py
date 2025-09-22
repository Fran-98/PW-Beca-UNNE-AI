from PW import PanguModel,get_model_path, load_era5_data
import torch
import torch.nn.functional as F
from torch.optim import Adam

def train(epochs=100, dataset_length=35000, learning_rate=5e-4, weight_decay=3e-6):
    print("Empezando entrenamiento")
    
    model = PanguModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    for i in range(epochs):
        model.train()
        total_loss = 0
        for step in range(dataset_length):
            # Normalizamos
            input_data, input_surface, target, target_surface = load_era5_data(step)
            input_data, input_surface = input_data.to(device), input_surface.to(device)
            target, target_surface = target.to(device), target_surface.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            output, output_surface = model(input_data, input_surface)

            # MAE loss conbias de 0.25 a superficie
            loss_upper = F.l1_loss(output, target)
            loss_surface = F.l1_loss(output_surface, target_surface)
            loss = loss_upper + 0.25 * loss_surface
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if step % 1000 == 0:
                print(f"  Epoch {i+1}/{epochs}, Step {step}/{dataset_length}, Loss: {loss.item():.6f}")

        print(f"Epoch {i+1}. Loss: {total_loss / dataset_length:.6f}")

    torch.save(model.state_dict(), get_model_path("final"))
    print("Entrenamiento listooo.")



if __name__ == '__main__':
    print("="*10)
    print("Testing modelo")
    print("="*10)
    
    try:
        model = PanguModel()
        # dummys
        dummy_input = torch.randn(1, 5, 13, 721, 1440) # B, C, P, H, W
        dummy_surface = torch.randn(1, 4, 721, 1440) # B, C, H, W
        
        output, output_surface = model(dummy_input, dummy_surface)
        
        # Check output shapes
        print(f"Sanity Check Passed!")
        print(f"Input upper shape: {dummy_input.shape}")
        print(f"Input surface shape: {dummy_surface.shape}")
        print(f"Output upper shape: {output.shape}")
        print(f"Output surface shape: {output_surface.shape}")
        assert output.shape == (1, 5, 13, 721, 1440)
        assert output_surface.shape == (1, 4, 721, 1440)
    except Exception as e:
        print(f"Sanity Check Failed: {e}")
