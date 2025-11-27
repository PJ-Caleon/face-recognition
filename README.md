# How to Use

## Step 1: Download Files
`git clone`

`cd face-recognition`

## Step 2: Upload images to folder dataset
- Make sure each image is `yourname.jpg`
- This is so that when testing the program, the model will take the `yourname` label to your specific facial features
- This is to allow multiple users to run the model
- The filename should be `yourname1.jpg` , `yourname2.jpg` , `yourname3.jpg` , `yourname4.jpg`...
- This is so the model can track what facial patterns it must consider before classifying the face and the number is there to serve as an ID to the program and a guide to how many images the user is using

## Step 3: Run the programs
- Run `python face_model.py` to train the model
- Run `python test_model.py` to test the model

## Step 4: Observe
The terminal will show how far off is the actual image to the images it was trained on. You may adjust the threshold if the face detection is too strict
