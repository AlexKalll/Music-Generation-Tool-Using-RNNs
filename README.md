# 🎶 Music Generation Using RNNs 🎹

Welcome 👋
I believe you will learn alot in this project.
 You can find [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) to train the model using piano midi files. Given a sequence of notes, the model learns to predict the next note in the sequence. 🎼

This project includes complete code to parse and create MIDI files. You can learn more about how RNNs work by visiting the [Text generation with an RNN](https://www.tensorflow.org/text/tutorials/text_generation) tutorial.

## 🎹 How It Works
1. Download the MAESTRO Dataset: The dataset contains approximately 1,200 MIDI files.
2. Process MIDI Files: Use pretty_midi to parse MIDI files and extract notes.
3. Create a Training Dataset: Extract notes from MIDI files and create a TensorFlow dataset.
4. Train the Model: Use an LSTM-based RNN to learn note sequences.
5. Generate Music: Use the trained model to generate new musical sequences!

## 🤝 Cloning the Repository and have a trials

To clone this repository to your local machine, follow these steps:

1. Open your terminal (or command prompt).
2. Navigate to the directory where you want to clone the repository.
3. Run the following command:
   ```bash
   git clone https://github.com/AlexKalll/Music-Generation-Using-RNNs.git

4. Navigate into the cloned directory:
    ```bash
    cd Music-Generation-Using-RNNs
5. You can create your own virtual environment for the sake of being safe for you global python interprator. 
- Create venv by typing the command below in your terminal.
    ```bash
    python -m venv myenv
- Activate it 
    ```bash 
    .\myenv\Scripts\activate

## 🚀 Setup

To get started, you'll need to install the following libraries:

1. **Install FluidSynth** (for audio playback):
   ```bash
   pip install -y fluidsynth
2. **Install Other Neccessary packages**
    ```bash
   pip install tensorflow pretty_midi pandas numpy matplotlib
#### 🎉 Finally, you are at a stage of running you model and get the out put music in the `output.midi` file. you can start playing with the piano classical music😊

###### Note that : - I highly recommend you to do this project in [google colab](https://colag.research.google.com), if you really interested in going through it.

# Huge Thanks to [CodeAlpha](https://codealpha.tech)! 🎉

I just wanted to take a moment to express my **heartfelt gratitude** to **[CodeAlpha](https://codealpha.tech)** for providing me with an incredible internship opportunity in the feilf of `AI/ML/DL`! 🙌✨

This experience has been nothing short of amazing, and I have learned so much! 📚💡
Thank you for believing in me and giving me the chance to grow both personally and professionally. 🌱💪

I’m truly grateful for this opportunity! Here’s to more learning and growth ahead! 🚀🌟

**Thank you, [CodeAlpha](https://codealpha.tech)!** 💖🙌🎉


### Get in touch with me
[LinkedIn](https://www.linkedin.com/in/kaletsidik-ayalew-mekonnen-34772226b/) | [Instagram](https://www.instagram.com/kaletsidik.24?igsh=YzljYTk1ODg3Zg==) | [X~Twitter](https://x.com/kaletsidike?t=VCe79O084EmE9bM2V5jOIA&s=09) | [Telegram](https://t.me/Adragon_de_mello) | [GitHub](https://github.com/AlexKalll) | [LeetCode](https://leetcode.com/Alexkal/)


Kaletsidik Ayalew
alexkalalw@gmail.com