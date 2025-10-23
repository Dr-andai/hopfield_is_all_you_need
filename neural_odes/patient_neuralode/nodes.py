import torch
import torch.nn as nn
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt

# Predicting whether a patient will detoriate in the next 24 hours

class PatientODE(nn.Module):
    """
    Neural ODE that captures patient’s physiological dynamics
    """
    def __init__(self, state_dim, hidden_dim):
        super(PatientODE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),   # 6 → 32
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),  # 32 → 32
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)    # 32 → 6
        )

    def forward(self, t, x):
        return self.net(x)

class PatientNeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PatientNeuralODE, self).__init__()
        self.ode_func = PatientODE(input_dim + output_dim, hidden_dim)
        self.initial_encoder = nn.Linear(input_dim, output_dim)
        self.output_layer = nn.Linear(input_dim + output_dim, output_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, initial_measurements, prediction_times):
        # encode intital measurements into hidden states
        hidden_init = self.initial_encoder(initial_measurements)

        # combine measurements and hidden states
        initial_state = torch.cat([initial_measurements, hidden_init], dim=1)
        
        # Solve ODE forward to get states at prediction times
        states = torchdiffeq.odeint(self.ode_func, initial_state, prediction_times, method='dopri5', rtol=1e-3, atol=1e-4)
        states = states.permute(1,0,2) # batch size, time points, features

        # predict detoriation probability
        outputs = torch.sigmoid(self.output_layer(states))

        return outputs, states
    
#### Simulate ICU data
def simulate_icu_data(num_patients = 100, max_time_points = 10):
    patient_data = []
    prediction_times = []
    labels = []

    for i in range(num_patients):
        num_measurements = np.random.randint(3, max_time_points)
        times = torch.cumsum(torch.randn(num_measurements)* 6, dim=0)
        times = times/ 24.0

        # Simulate vital signs:[heart_rate, bp_systolic, bp_diastolic, spo2, temperature]
        base_vitals = torch.tensor([80.0, 120.0, 80.0, 98.0, 37.0])

        measurements = []
        for t in times:
            # time depenednt variation and noise
            trend = torch.tensor([t * 10, -t*20, -t*10, -t*2, t*0.5])
            noise = torch.randn(5) * torch.tensor([5.0, 10.0, 5.0, 1.0, 0.3])
            vitals = base_vitals + trend + noise
            measurements.append(vitals)

        measurements = torch.stack(measurements)

        # Simulate detoriation label
        hr_trend = measurements[-1,0] - measurements[0,0]
        spo2_trend = measurements[-1,3] - measurements[0,3]
        will_detoriate = 1 if (hr_trend > 15 and spo2_trend <-2) else 0

        patient_data.append(measurements)
        prediction_times.append(times)
        labels.append(will_detoriate)

    return patient_data, prediction_times, labels


### Training loop
def train_patient_model():
    ## Model paramters
    input_dim = 5 # [HR, BP_sys, BP_dia, SPO2, Temp]
    hidden_dim = 32
    output_dim = 1

    model = PatientNeuralODE(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # simulate training data
    train_data, train_times, train_labels = simulate_icu_data(500)

    for epoch in range(100):
        total_loss = 0
        for i in range(len(train_data)):
            initial_measurement = train_data[i][0:1]
            times = train_times[i]

            # We want to predict at the last final time
            prediction_time = times[-1:]

            # Forward pass
            predictions, _ = model(initial_measurement, prediction_time)
            final_prediction = predictions[0,-1,0] # prediction at the last time

            # Loss
            true_label = torch.tensor([train_labels[i]], dtype = torch.float32)
            loss = criterion(final_prediction.unsqueeze(0), true_label)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}/len{train_data}:.4f")
    return model

### Visualization
def visualize_patient_trajectory(model, patient_data, patient_times):
    """
    Visualize how model sees patient evolution
    """
    initial_measurement = patient_data[0:1]
    times_continous = torch.linspace(0, 1.0, 100) # continous time points

    with torch.no_grad():
        predictions, states = model(initial_measurement, times_continous)

    # plot continous trajectory
    plt.figure(figsize=(12, 4))

    # Plt heart rate trajectory
    plt.subplot(1,2,1)
    hr_trajectory = states[0,:,0].numpy() # heart_rate component
    plt.plot(times_continous.numpy(), hr_trajectory,'b-', alpha= 0.7, label='Learned HR trajectory')

    # Plot actual measurements
    actual_times = patient_times
    actual_hr = patient_data[:,0].numpy()
    plt.scatter(actual_times.numpy(), actual_hr, color='red', s=50, zorder=5, label='Actual measurements')

    plt.xlabel('Time (normalized)')
    plt.ylabel('Heart Rate')
    plt.legend()
    plt.title('Continous Patient Trajectory')

    # Plot detoriation probability
    plt.subplot(1, 2, 2)
    prob_trajectory = predictions[0,:,0].numpy()
    plt.plot(times_continous.numpy(), prob_trajectory,'r-', linewidth=2)
    plt.xlabel('Time(normalized)')
    plt.ylabel('Detoriatio probability')
    plt.title('Risk prediction over time')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

# Train_test
if __name__ == "__main__":
    print("Training Neural ODE for patient detoriation prediction...")
    model = train_patient_model()

    # test on a sample patient
    test_data, test_times, test_labels = simulate_icu_data(1)
    print('\nSample patient - Actual detoriation: {test_labels[0]}')

    with torch.no_grad():
        initial_measurement = test_data[0][0:1]
        prediction_time = test_times[0][-1:]
        prediction,_ = model(initial_measurement, prediction_time)
        print("Predicted detoriation probability: {prediction[0, -1,0]:.3f}")

    visualize_patient_trajectory(model, test_data[0], test_times[0]) 