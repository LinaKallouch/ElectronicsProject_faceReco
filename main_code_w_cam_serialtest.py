import os
import numpy as np
import face_recognition
import cv2
import time
import pyrealsense2 as rs

class FaceMatcher:
    def __init__(self, assistants_folder: str, compare_duration: int = 6):
        self.assistants_folder = assistants_folder
        self.compare_duration = compare_duration
        self.best_match = None
        self.best_score = 0
        self.assistant_scores = {}
        self.selection_count = 0
        self.score_toren = 0
        self.nieuwe_current_position_toren = None
        self.hoogte_toren = [0, 220, 400, 650, 900]
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
    
    def compare_faces(self, reference_encoding: np.ndarray, face_encoding: np.ndarray):
        face_distance = face_recognition.face_distance([reference_encoding], face_encoding)[0]
        similarity_score = (1 - face_distance) * 100  # afstand ligt tussen 0 en 1 bij 0-> 100% idem persoon bij 1-> max afstand 
                                                      #in de ruimte en dus heb je 0% gelijkenis met die persoon
        return similarity_score
    
    def detect_faces(self, image: np.ndarray):       #input: een afbeelding als een NumPy-array van pixels
                                                     #output: coördinaten van gezichten in de afbeelding
                                                     #hoe? een CNN herkent patronen en trekt een bounding box rond een gezicht
        return face_recognition.face_locations(image)
    
    def extract_features(self, image: np.ndarray, face_locations): #output: Een 128D vector per gezicht
                                                                   #hoe? De CNN-laag van face_recognition berekent een unieke gezichtsencoding
        return face_recognition.face_encodings(image, known_face_locations=face_locations)
    
    def load_assistant_encodings(self):
        assistant_encodings = {}
        for assistant_folder in os.listdir(self.assistants_folder):
            folder_path = os.path.join(self.assistants_folder, assistant_folder)
            if not os.path.isdir(folder_path):
                continue
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
                if encodings:
                    assistant_encodings[assistant_folder] = encodings[0]
                    break
        return assistant_encodings
    
    def capture_and_compare(self):
        assistants = self.load_assistant_encodings()
        self.best_score = 0
        self.best_match = None
        for assistant_name, assistant_encoding in assistants.items():
            start_time = time.time()
            best_score = 0
            while time.time() - start_time < self.compare_duration:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                face_locations = self.detect_faces(rgb_frame)
                encodings = self.extract_features(rgb_frame, face_locations)
                for face_encoding in encodings:
                    similarity = self.compare_faces(assistant_encoding, face_encoding)
                    if similarity > best_score:
                        best_score = similarity
                cv2.putText(color_image, f"Comparing with: {assistant_name}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(color_image, f"Score: {best_score:.2f}%", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Face Comparison", color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.assistant_scores[assistant_name] = best_score
            print(f"{assistant_name}: {best_score:.2f}% overeenkomst.")
            if best_score > self.best_score:
                self.best_score = best_score
                self.best_match = assistant_name
        cv2.destroyAllWindows()
        
    def get_selected_assistant(self, selected_string):
        try:
            selected_index = int(selected_string.split(": ")[1]) - 1 #["Selected", "2"]
            assistants = sorted(os.listdir(self.assistants_folder)) 
            if selected_index < len(assistants):
                return assistants[selected_index]
            else:
                return None
        except:
            return None #als er een fout optreedt (bv als de string niet correct is)

    def verify_selected_assistant(self, selected_string):
        selected_assistant = self.get_selected_assistant(selected_string)
        if not selected_assistant:
            print("Ongeldige assistent of geen score gevonden.")
            return

        selected_score = self.assistant_scores.get(selected_assistant, None) #dict.get() met default value kon 0 zn
        print(f"Geselecteerde assistent: {selected_assistant} en had een score van {selected_score:.2f}%.")

        self.update_score(selected_assistant) # fie opgeroepen hieronder
    def update_score(self, selected_assistant):
        nieuwe_current_position_toren = self.hoogte_toren[self.score_toren]
        if self.best_match == selected_assistant:
            
            if self.score_toren < len(self.hoogte_toren) - 1: # -1 want zodat we bij elke selected een index vanaf 0 kunnen koppelen
                self.score_toren += 1
                self.nieuwe_current_position_toren = self.hoogte_toren[self.score_toren]
                
                print('Current positie toren:', self.nieuwe_current_position_toren)
                print(f"Score verhoogd, met 1 punt dankzij Player{self.selection_count +1}!")
            
        else:
            print(f"Geen punten, de beste match komt niet overeen met de selectie, huidige score: {self.score_toren} op positie {nieuwe_current_position_toren} in toren.")

    def game_loop(self):
        test_serial_inputs = ["Selected: 5", "Selected: 5", "Selected: 1", "Selected: 5"]
        while self.selection_count < 4:
            print(f"Player{self.selection_count +1}, kijk naar de camera we starten met de gezichtsherkenning!")
            self.capture_and_compare()
            if test_serial_inputs:
                selected_string = test_serial_inputs.pop(0)
                self.verify_selected_assistant(selected_string)
                self.selection_count += 1
            time.sleep(2)
        print(f"Game over! Final score: {self.score_toren}/4 en zit op op hoogte: {self.nieuwe_current_position_toren}!")
        self.pipeline.stop()

assistants_folder = "All Pictures"
matcher = FaceMatcher(assistants_folder)
matcher.game_loop()
