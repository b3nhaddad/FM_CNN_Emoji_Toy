# Flow Matching w/ Convolutional Neural Network on Emoji Dataset

<img src="public_repo_images/woman_on_fire.png" alt="Woman on Fire" width="400" height="auto">
<p style="color: aqua; color-scheme: light; accent-color: aqua;">from prompt 'Woman on fire generated' by the model</p>

Flow Matching via CNN (ResNet architecture, with CLIP word embeddings, training on apple's 'emoji' (therefore cannot be used commercially)

The first two iterations opted to use grayscale imaging as I thought it would be less computationally expensive (as well as using float base 32 instead of 64)

I increased to RGB later since using grayscale proved harder to find a velocity field for the pixels: which would increase R^n*3 (for each color channel)

The 'flow' of this training loop is to take the emoji static word embeddings (these are pre-generated and stored in the emoji_embeddings.pt file),
these embeddings are grouped with the image pairs that are created 
then generate an interpolated sample between the image X_1 and 
an image of gaussian 'noise' (each pixel is sampled from the distribution w/ sigma = 0 & variance of the identity matrix).
This gives us the OTP (optimal transport path) between the image and noise at a random timestep (drawn from a uniform distribution of the size
defined by your discretized time space).

