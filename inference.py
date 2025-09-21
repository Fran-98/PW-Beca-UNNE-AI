from PW import PanguModel, load_static_data, load_era5_data
import torch

def inference(input_data, input_surface, forecast_range=20):
    """
    Inferencencia con diferentes lead times.
    Sería para el PW original con modelos con delta de 1, 3, 6, y 24 horas.
    """
    print("Comenzandooo...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Por ahora utilizo como placeholder el mismo modelo vacio, aca tienen un LoadModel originalmente
    pangu_model_24 = PanguModel().to(device).eval()
    pangu_model_6 = PanguModel().to(device).eval()
    pangu_model_3 = PanguModel().to(device).eval()
    pangu_model_1 = PanguModel().to(device).eval()

    # De la función de carga de datos sacamos mean y std
    weather_mean, weather_std, surf_mean, surf_std = load_static_data()

    # Todo a gpu
    weather_mean, weather_std = weather_mean.to(device), weather_std.to(device)
    surf_mean, surf_std = surf_mean.to(device), surf_std.to(device)

    # Estados iniciales para diferentes modelos
    input_24, input_surface_24 = input_data.clone().to(device), input_surface.clone().to(device)
    input_6, input_surface_6 = input_data.clone().to(device), input_surface.clone().to(device)
    input_3, input_surface_3 = input_data.clone().to(device), input_surface.clone().to(device)
    
    current_input, current_surface_input = input_data.clone().to(device), input_surface.clone().to(device)

    output_list = []
    
    with torch.no_grad():
        for i in range(forecast_range):
            lead_time = i + 1
            print(f"Prediciendo para un delta de {lead_time}...")
            
            # Acá seleccionamos el modelo mas adecuado
            if lead_time % 24 == 0:
                print("Usando modelo de 24 horas")
                current_input, current_surface_input = input_24, input_surface_24
                output, output_surface = pangu_model_24(current_input, current_surface_input)
                # Store output for the next 24-hour step
                input_24, input_surface_24 = output.clone(), output_surface.clone()
                input_6, input_surface_6 = output.clone(), output_surface.clone()
                input_3, input_surface_3 = output.clone(), output_surface.clone()
            elif lead_time % 6 == 0:
                print("Usando modelo de 6 horas")
                current_input, current_surface_input = input_6, input_surface_6
                output, output_surface = pangu_model_6(current_input, current_surface_input)
                # Store output for the next 6-hour step
                input_6, input_surface_6 = output.clone(), output_surface.clone()
                input_3, input_surface_3 = output.clone(), output_surface.clone()
            elif lead_time % 3 == 0:
                print("Usando modelo de 3 horas")
                current_input, current_surface_input = input_3, input_surface_3
                output, output_surface = pangu_model_3(current_input, current_surface_input)
                # Store output for the next 3-hour step
                input_3, input_surface_3 = output.clone(), output_surface.clone()
            else:
                print("Usando modelo de 1 hora")
                output, output_surface = pangu_model_1(current_input, current_surface_input)

            # Actualizamos el input para el proximo paso
            current_input, current_surface_input = output, output_surface

            # Desnormalizamos
            output_denorm = output * weather_std + weather_mean
            output_surface_denorm = output_surface * surf_std + surf_mean
            
            output_list.append((output_denorm.cpu(), output_surface_denorm.cpu()))
            
    print(f"INFERENCIA COMPLETAA. Se generaron {len(output_list)} predicciones.")
    return output_list


if __name__ == "__main__":
    print("\n" + "="*10)
    print("Inferencia de prueba")
    print("="*10)
    initial_input, initial_surface, _, _ = load_era5_data(0)
    forecasts = inference(initial_input, initial_surface, forecast_range=4)