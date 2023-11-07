import json
from mtcnn.mtcnn import MTCNN
import cv2
import os
from multiprocessing import Pool

splash = ("""
======================================================
    __  ___                   ______                  
   /  |/  /___ ___________   / ____/________  ____    
  / /|_/ / __ `/ ___/ ___/  / /   / ___/ __ \/ __ \     
 / /  / / /_/ (__  |__  )  / /___/ /  / /_/ / /_/ /
/_/  /_/\__,_/____/____/   \____/_/   \____/ .___/ 
                                          /_/      
=======================================================
        """)


def clear():
    name = os.name
    if name == 'nt':  # windows
        os.system('cls')
    else:  # mac / linux
        os.system('clear')


def save_settings(file_path, settings):
    with open(file_path, 'w') as json_file:
        json.dump(settings, json_file, indent=4)
        clear()
        print("Saved")


def display_settings(settings):
    print("""
========
Settings
========    
    """)
    print("[1] Padding: " + str(settings['padding']))
    print("[2] Input file path: " + settings['path'])
    print("[3] Output file path: " + settings['out'])
    print("[4] Threads: " + str(settings['threads']))
    print("""
========
[5] Menu    
    """)


def settings_menu(file_path, settings):
    while True:
        display_settings(settings)
        choice = input("Select setting (1/2/3/4/5): ")

        if choice == '1':
            new_padding = input("Enter new padding: ")
            try:
                new_padding = int(new_padding)
                settings['padding'] = new_padding
                save_settings(file_path, settings)
            except ValueError:
                print("Invalid input. Please enter a valid integer for padding.")

        elif choice == '2':
            new_path = input("Enter new input path: ")
            settings['path'] = new_path
            save_settings(file_path, settings)

        elif choice == '3':
            new_out = input("Enter new output file path: ")
            settings['out'] = new_out
            save_settings(file_path, settings)

        elif choice == '4':
            new_threads = input("How many concurrent instances: ")
            try:
                new_threads = int(new_threads)
                settings['threads'] = new_threads
                save_settings(file_path, settings)
            except ValueError:
                print("Invalid input. Please enter a valid integer for threads.")

        elif choice == '5':
            break

        else:
            print("Please enter a valid choice, [1-5]")



def init_settings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            settings = json.load(json_file)

    else:
        settings = {
            'padding': 470,
            'path': '',  # Initialize path as an empty string
            'out': '',
            'threads': 2,  # Initialize threads with a default value (e.g., 2)
        }
        with open(file_path, 'w') as json_file:
            json.dump(settings, json_file, indent=4)

        # If path is empty, prompt the user to set it
        while not settings['path']:
            settings['path'] = input("Enter the input path: ")

        # Save the updated settings with the path
        with open(file_path, 'w') as json_file:
            json.dump(settings, json_file, indent=4)

    return settings

settings = init_settings('settings.json')
detector = MTCNN()
path = settings['path']
out = settings['out']
padding = settings['padding']
threads = settings['threads']


# Image processing function
def cook_image(image_file):
    print(f"Processing Image: {image_file}")
    image_path = os.path.join(path, image_file)
    image = cv2.imread(image_path)
    cooked_image = cv2.cvtColor(image,
                                cv2.COLOR_BGR2RGB)  # Processed image is greyscale unless dimensions are specified by
    # the user - abstracting colours improves accuracy in detection significantly
    return cooked_image


def process_cropped_images(images, cooked_images):
    num_processes = int(threads)
    with Pool(num_processes) as pool:
        cropped_faces = pool.starmap(crop_save, zip(images, cooked_images))

    return cropped_faces


def crop_save(image, cooked_image):
    faces = detector.detect_faces(cooked_image)
    faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)

    if len(faces) > 0:
        x, y, w, h = faces[0]['box']

        # Adjust bounding box
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding

        # Ensure bounding box stays within image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

        # Crop ROI from the original image
        ROI = image[y:y + h, x:x + w]
        return ROI


def process_images(image_files):
    num_processes = int(threads)
    print(f"Processing Image: {image_files} - Process ID: {os.getpid()}")
    with Pool(num_processes) as pool:
        cooked_images = pool.map(cook_image, image_files)

    cropped_faces = process_cropped_images([cv2.imread(os.path.join(path, f)) for f in image_files], cooked_images)

    for image_file, cropped_face in zip(image_files, cropped_faces):
        output_path = os.path.join(out, f"cropped_{image_file}")
        cv2.imwrite(output_path, cropped_face)

        print(f"{image_file} Processed")


def main_menu():
    try:
        while True:
            clear()
            print(splash)
            print("[1] Start job in > " + settings['path'])
            print("[2] Settings")
            print("[3] Quit")

            choice = input("Enter your choice (1/2/3): ")

            if choice == '1':
                clear()
                image_files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png', 'JPG'))]
                process_images(image_files)
            elif choice == '2':
                clear()
                settings_menu('settings.json', settings)
            elif choice == '3':
                print("Quitting the program.")
                return True
            elif choice.lower() == "menu":
                continue
            else:
                print("Invalid choice. Please select a valid option.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    while main_menu() is not True:
        pass
