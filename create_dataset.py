import pandas as pd
import random

def create_synthetic_data(output_path="data/dataset.csv", num_samples=100):
    genres = ["Science Fiction", "Romance", "Mystery", "Fantasy"]
    
    data = []
    for _ in range(num_samples):
        genre = random.choice(genres)
        if genre == "Science Fiction":
            text = "The spaceship landed on the alien planet. The robots were friendly."
        elif genre == "Romance":
            text = "She looked into his eyes and knew he was the one. Love was in the air."
        elif genre == "Mystery":
            text = "The detective found a clue at the crime scene. Who was the killer?"
        elif genre == "Fantasy":
            text = "The dragon flew over the castle. The wizard cast a spell."
            
        # Add some random noise
        text += " " + " ".join([random.choice(["the", "a", "is", "was"]) for _ in range(3)])
        
        data.append({"text": text, "genre": genre})
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Created synthetic dataset at {output_path}")

if __name__ == "__main__":
    create_synthetic_data()
