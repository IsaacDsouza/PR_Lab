import cv2
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# Define fuzzy variables
brightness = ctrl.Antecedent(np.arange(0, 256, 1), 'brightness')
edge_intensity = ctrl.Antecedent(np.arange(0, 256, 1), 'edge_intensity')
classification = ctrl.Consequent(np.arange(0, 101, 1), 'classification')

# Membership functions
brightness.automf(3, names=['dark', 'normal', 'bright'])
edge_intensity.automf(3, names=['low', 'medium', 'high'])
classification.automf(3, names=['low', 'medium', 'high'])

# Define rules
rules = [
    ctrl.Rule(brightness['dark'] | edge_intensity['low'], classification['low']),
    ctrl.Rule(brightness['normal'] | edge_intensity['medium'], classification['medium']),
    ctrl.Rule(brightness['bright'] | edge_intensity['high'], classification['high'])
]

# Create fuzzy control system
classifier = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

def real_time_image_classification():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        classifier.input['brightness'] = np.mean(gray)
        classifier.input['edge_intensity'] = np.mean(cv2.Canny(gray, 100, 200))
        classifier.compute()
        
        cv2.putText(frame, f'Classification: {classifier.output["classification"]:.2f}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Real-Time Classification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_image_classification()
