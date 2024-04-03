# Inspiration
Did you know that 15 million Americans had their identity stolen in 2021? Our team's inspiration for AuthentInk came from this shared passion for combating identity fraud in the digital age. With the increasing reliance on digital signatures for important transactions and documents, we recognized the need for a robust and reliable signature forgery detector. We aimed to create a solution that could protect individuals and organizations from the growing threat of signature forgery.

## What it does
AuthentInk is a cutting-edge signature forgery detector powered by Siamese Neural Networks (Siamese NN) and Convolutional Neural Networks (CNN). It works by comparing the input signature against a reference database, verifying its authenticity in real-time. Here's what it does:

- Signature Input: Users can capture a signature image through a PyQt5 app
- Authentication: AuthentInk compares the input signature to a pre-trained dataset of genuine signatures. It calculates the similarity score using Siamese NN
- Result: AuthentInk determines whether the signature is genuine or potentially forged based on the similarity score.

## How we built it
We used Kaggle to retrieve the data Preprocessed the data by making them uniform Training through Siamese NN and Contrastive loss to evaluate similarity score PyQt5 to create the UI and the implementation of the camera

## Challenges we ran into
Being complete newcomers to AI, implementing Siamese neural networks, convolutional neural networks, and contrastive loss proved highly challenging, leading us to encounter many bugs and making us look for extensive online research and guidance.

## Accomplishments that we're proud of
The project is fully functional, and it can detect fairly accurately if a signature is fraudulent

## What we learned
SiameseNN, Convolution, Numpy, Contrastive Loss, Processing Data/Images, Pytorch, PyQt5, OpenCV (camera)

## What's next for AuthentInk
Bank verification More data to have higher precision and accuracy

## Built With
- numpy
- opencv
- pyqt5
- python
- pytorch
- siamesenn
