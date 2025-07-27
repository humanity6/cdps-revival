# Known Faces Directory

This directory contains face images for the crime detection system.

## Directory Structure

- `restricted/` - Images of people with restricted access
- `criminals/` - Images of known criminals or suspects

## Adding Faces

1. **Image Format**: Use JPG, JPEG, PNG, or BMP files
2. **Image Quality**: Clear, front-facing photos work best
3. **File Naming**: Use the person's name as the filename (e.g., `john_doe.jpg`)
4. **One Face Per Image**: Ensure only one face is visible in each image

## Example Usage

```
known_faces/
├── restricted/
│   ├── employee_john.jpg
│   ├── visitor_jane.jpg
│   └── contractor_mike.jpg
└── criminals/
    ├── suspect_001.jpg
    ├── wanted_person.jpg
    └── criminal_abc.jpg
```

## System Commands

To rebuild the face database after adding new images:
```bash
python face_detection_system.py --rebuild-db
```

To run the system:
```bash
# Webcam detection
python face_detection_system.py --source webcam

# Video file detection
python face_detection_system.py --source path/to/video.mp4
```