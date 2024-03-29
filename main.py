import numpy as np
from scipy.io.wavfile import write, read
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QDialog, \
    QTabWidget, QComboBox, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import find_peaks, butter, lfilter, freqz,filtfilt


class CharacterEncoder:
    def __init__(self):
        # Dictionary of characters mapped to their frequency tuples
        self.character_frequencies = {
            'a': (100, 1100, 2500),
            'b': (100, 1100, 3000),
            'c': (100, 1100, 3500),
            'd': (100, 1300, 2500),
            'e': (100, 1300, 3000),
            'f': (100, 1300, 3500),
            'g': (100, 1500, 2500),
            'h': (100, 1500, 3000),
            'i': (100, 1500, 3500),
            'j': (300, 1100, 2500),
            'k': (300, 1100, 3000),
            'l': (300, 1100, 3500),
            'm': (300, 1300, 2500),
            'n': (300, 1300, 3000),
            'o': (300, 1300, 3500),
            'p': (300, 1500, 2500),
            'q': (300, 1500, 3000),
            'r': (300, 1500, 3500),
            's': (500, 1100, 2500),
            't': (500, 1100, 3000),
            'u': (500, 1100, 3500),
            'v': (500, 1300, 2500),
            'w': (500, 1300, 3000),
            'x': (500, 1300, 3500),
            'y': (500, 1500, 2500),
            'z': (500, 1500, 3000),
            ' ': (500, 1500, 3500)
        }

    def encode_character(self, char):
        # Returns the frequency tuple for a given character or (0, 0, 0) if not found

        return self.character_frequencies.get(char, (0, 0, 0))

    def encode_string(self, input_str):

        # Encodes a string into a concatenated signal of frequency components
        encoded_signal = []
        for char in input_str.lower():
            frequencies = self.encode_character(char)
            encoded_signal.extend(self.generate_signal(frequencies))
        return np.array(encoded_signal)

    def generate_signal(self, frequencies, duration=0.04, sample_rate=8000):
        # Generates a signal based on the given frequencies
        t = np.linspace(0, duration, int(
            sample_rate * duration), endpoint=False)
        # print(t)
        signal = np.sin(2 * np.pi * frequencies[0] * t) + \
            np.sin(2 * np.pi * frequencies[1] * t) + \
            np.sin(2 * np.pi * frequencies[2] * t)
        return signal

    def generate_dft(self, signal, sample_rate=8000):
        # Computes the Discrete Fourier Transform (DFT) of a signal
        dft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(dft), 1 / sample_rate)
        return freqs, np.abs(dft)


class CharacterDecoder:
    def __init__(self, sample_rate=8000, duration=0.04):
        # Initialize decoder with sample rate and duration
        self.sample_rate = sample_rate
        self.duration = duration
        # Dictionary of characters mapped to their frequency tuples
        self.character_frequencies = {
            'a': (100, 1100, 2500),
            'b': (100, 1100, 3000),
            'c': (100, 1100, 3500),
            'd': (100, 1300, 2500),
            'e': (100, 1300, 3000),
            'f': (100, 1300, 3500),
            'g': (100, 1500, 2500),
            'h': (100, 1500, 3000),
            'i': (100, 1500, 3500),
            'j': (300, 1100, 2500),
            'k': (300, 1100, 3000),
            'l': (300, 1100, 3500),
            'm': (300, 1300, 2500),
            'n': (300, 1300, 3000),
            'o': (300, 1300, 3500),
            'p': (300, 1500, 2500),
            'q': (300, 1500, 3000),
            'r': (300, 1500, 3500),
            's': (500, 1100, 2500),
            't': (500, 1100, 3000),
            'u': (500, 1100, 3500),
            'v': (500, 1300, 2500),
            'w': (500, 1300, 3000),
            'x': (500, 1300, 3500),
            'y': (500, 1500, 2500),
            'z': (500, 1500, 3000),
            ' ': (500, 1500, 3500)
        }

    def decode_frequency_analysis(self, signal):
        # Decode signal using frequency analysis method
        decoded_string = ""
        for i in range(0, len(signal), int(self.sample_rate * self.duration)):
            chunk = signal[i:i + int(self.sample_rate * self.duration)]
            frequencies = self.frequency_analysis(chunk)
            char = self.find_char(frequencies)
            decoded_string += char

        return decoded_string
   
    def design_bandpass_filters(self, bandwidth=5.0):
        filters = {}

        for char, char_frequencies in self.character_frequencies.items():
            if isinstance(char_frequencies, (list, tuple)) and len(char_frequencies) == 3:
                lowcut, middle, highcut = char_frequencies
            else:
                raise ValueError(f"Invalid format for char_frequencies: {char_frequencies}")

            nyquist = 0.5 * self.sample_rate

            # Design lowpass filter
            low = (lowcut - bandwidth/2) / nyquist
            high = (lowcut + bandwidth/2) / nyquist
            b_low, a_low = butter(5, [low, high], btype='band')
            filters[char] = {'b_low': b_low, 'a_low': a_low}

            # Plot lowpass filter
           # self.plot_filter_response(b_low, a_low, char, 'Lowpass Filter')

            # Design bandpass filter for middle frequency
            low = (middle - bandwidth/2) / nyquist
            high = (middle + bandwidth/2) / nyquist
            b_middle, a_middle = butter(5, [low, high], btype='band')
            filters[char]['b_middle'] = b_middle
            filters[char]['a_middle'] = a_middle

            # Plot bandpass filter for middle frequency
            #self.plot_filter_response(b_middle, a_middle, char, 'Bandpass Filter (Middle)')

            # Design highpass filter
            low = (highcut - bandwidth/2) / nyquist
            high = (highcut + bandwidth/2) / nyquist
            b_high, a_high = butter(5, [low, high], btype='band')
            filters[char]['b_high'] = b_high
            filters[char]['a_high'] = a_high

            # Plot highpass filter
            #self.plot_filter_response(b_high, a_high, char, 'Highpass Filter')

        return filters

    def plot_filter_response(self, b, a, char, filter_type):
        w, h = freqz(b, a)
        plt.figure()  # Create a new figure for each plot
        plt.plot(0.5 * self.sample_rate * w / np.pi, np.abs(h), label=filter_type)
        plt.title(f'Frequency Response for {filter_type} - Character {char}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain')
        plt.legend()
        plt.grid(True)
        plt.show()
        
   


    def apply_bandpass_filters(self, signal, filters, chunk_size=0.04):
        # Apply bandpass filters to the input signal
        filtered_signals = {}
        decoded_string = ""

      # Split the signal into chunks of chunk_size seconds
        for i in range(0, len(signal), int(self.sample_rate * chunk_size)):
            chunk = signal[i:i + int(self.sample_rate * chunk_size)]
            frequencies = self.frequency_analysis(chunk)
            char = self.find_char(frequencies)
            decoded_string += char


            for char, filter_coeffs in filters.items():
                # Extract filter coefficients for the current character
                b_low, a_low = filter_coeffs['b_low'], filter_coeffs['a_low']
                b_middle, a_middle = filter_coeffs['b_middle'], filter_coeffs['a_middle']
                b_high, a_high = filter_coeffs['b_high'], filter_coeffs['a_high']

                # Get the frequency response for each filter
                w_low, h_low = freqz(b_low, a_low, worN=1000)
                w_middle, h_middle = freqz(b_middle, a_middle, worN=1000)
                w_high, h_high = freqz(b_high, a_high, worN=1000)

                # Find the index corresponding to the closest frequency in the frequency response
                index_low = np.argmin(np.abs(w_low - frequencies[1]))
                index_middle = np.argmin(np.abs(w_middle - frequencies[2]))
                index_high = np.argmin(np.abs(w_high - frequencies[0]))

                # Check if the center frequencies of filters match the frequencies of the chunk
                if np.isclose(h_low[index_low], 1.0) and np.isclose(h_middle[index_middle], 1.0) and np.isclose(h_high[index_high], 1.0):
                    # Apply the bandpass filters to the chunk
                    filtered_low = filtfilt(b_low, a_low, chunk)
                    filtered_middle = filtfilt(b_middle, a_middle, chunk)
                    filtered_high = filtfilt(b_high, a_high, chunk)

                  

                    # Store the filtered signals for each character and chunk
                    if char not in filtered_signals:
                        filtered_signals[char] = {'signal_low': [], 'signal_middle': [], 'signal_high': []}

                    filtered_signals[char]['signal_low'].append(filtered_low)
                    filtered_signals[char]['signal_middle'].append(filtered_middle)
                    filtered_signals[char]['signal_high'].append(filtered_high)

        return decoded_string

   
   
    def decode_bandpass_filters(self, signal):
        # Design bandpass filters
        decoded_string = ""
        filters = self.design_bandpass_filters()
        # Apply bandpass filters to the input signal
        decoded_string  = self.apply_bandpass_filters(signal, filters)
        return decoded_string

        

    def plot_frequency_response(self, b, a, char):
          # Plot the frequency response for each filter
            # self.plot_frequency_response(b_low, a_low, char + '_low')
            # self.plot_frequency_response(b_middle, a_middle, char + '_middle')
            # self.plot_frequency_response(b_high, a_high, char + '_high')
        # Plot the frequency response of the filter
        w, h = freqz(b, a, worN=8000)
        plt.figure()
        plt.plot(0.5 * self.sample_rate * w / np.pi, np.abs(h), 'b')
        plt.title(f'Frequency Response for character "{char}"')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain')
        plt.grid()
        plt.show()


    
    def frequency_analysis(self, chunk, plot=True):
        # Analyze frequencies in the input chunk
        spectrum = np.fft.fft(chunk)
        freqs = np.fft.fftfreq(len(chunk), 1 / self.sample_rate)
        positive_freqs = freqs[freqs > 0]
        positive_spectrum = spectrum[freqs > 0]
        peaks, _ = find_peaks(np.abs(positive_spectrum), height=0)
        sorted_peaks = sorted(peaks, key=lambda x: np.abs(
            positive_spectrum[x]), reverse=True)
        highest_peaks = sorted_peaks[:3]
        pulse_frequencies = positive_freqs[highest_peaks]

        return pulse_frequencies

    def find_nearest_character(self, target_frequency):
        # Find the nearest character based on target frequency
        min_distance = float('inf')
        nearest_char = 'unknown_char'

        for char, char_frequencies in self.character_frequencies.items():
            # Calculate the distance between target frequency and each character's frequencies
            distance = np.linalg.norm(
                np.array(char_frequencies) - target_frequency)

            # Update nearest character if the current distance is smaller
            if distance < min_distance:
                min_distance = distance
                nearest_char = char

        return nearest_char

    def find_char(self, target_tuple):
        # Find the character based on the target tuple
        sorted_target_tuple = tuple(sorted(target_tuple))
        sorted_target_tuple_int = tuple(int(element)
                                        for element in sorted_target_tuple)
        for char, char_tuple in self.character_frequencies.items():
            if char_tuple == sorted_target_tuple_int:
                return char
        return ''  # Replace 'unknown_char' with the default character you want to use


class SignalViewer(QDialog):
    def __init__(self, signal):
        # Initialize the SignalViewer with the provided signal
        super().__init__()
        self.signal = signal
        self.init_ui()

    def init_ui(self):
        # Set up the UI for the SignalViewer
        self.setWindowTitle('Generated Signal Viewer')

        # Create a matplotlib figure and canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # Plot the signal
        self.ax.plot(self.signal)
        self.ax.set_title('Generated Signal')
        self.ax.set_xlabel('Sample')
        self.ax.set_ylabel('Amplitude')

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)


class EncoderDecoderGUI(QWidget):
    decoded_result = None

    def __init__(self):
        super().__init__()
        self.encoder = CharacterEncoder()
        self.decoder = CharacterDecoder()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Character Encoder and Decoder GUI')
        self.setGeometry(300, 300, 600, 400)

        self.tab_widget = QTabWidget()

        # Encoding Tab
        self.encoding_tab = QWidget()
        self.init_encoding_tab()
        self.tab_widget.addTab(self.encoding_tab, 'Encoding')

        # Decoding Tab
        self.decoding_tab = QWidget()
        self.init_decoding_tab()
        self.tab_widget.addTab(self.decoding_tab, 'Decoding')

        layout = QVBoxLayout()
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

    def init_encoding_tab(self):
        # Set up the UI for the Encoding Tab
        self.input_label = QLabel('Enter English String:')

        self.input_text = QLineEdit(self)
        self.encode_button = QPushButton('Encode', self)
        self.play_button = QPushButton('Play Signal', self)
        self.save_button = QPushButton('Save Signal', self)
        self.view_button = QPushButton('View Signal', self)
        self.dft_button = QPushButton('Plot DFT', self)

        layout = QVBoxLayout()
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_text)
        layout.addWidget(self.encode_button)
        layout.addWidget(self.play_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.view_button)
        layout.addWidget(self.dft_button)

        self.encode_button.clicked.connect(self.encode_string)
        self.play_button.clicked.connect(self.play_signal)
        self.save_button.clicked.connect(self.save_signal)
        self.view_button.clicked.connect(self.view_signal)
        self.dft_button.clicked.connect(self.plot_dft)

        self.encoding_tab.setLayout(layout)

    def init_decoding_tab(self):
        # Set up the UI for the Decoding Tab
        self.result_label = QLabel('Decoded Result:')
        self.result_text = QLineEdit(self)
        self.result_acc = QLabel('Accuracy Result:')
        self.res_acc = QLineEdit(self)
        self.upload_button = QPushButton('Upload Audio File', self)
        self.run_button = QPushButton('Run Decoder', self)
        self.plot_button = QPushButton('Plot Signal', self)
        self.Measure_Acc = QPushButton('Measure Accuracy', self)
        self.exit_button = QPushButton('Exit', self)

        self.decode_method_label = QLabel('Select Decoding Method:')
        self.decode_method_dropdown = QComboBox()
        self.decode_method_dropdown.addItems(
            ['Bandpass Filters', 'Frequency Analysis (DFT)'])

        layout = QVBoxLayout()
        layout.addWidget(self.decode_method_label)
        layout.addWidget(self.decode_method_dropdown)
        layout.addWidget(self.result_label)
        layout.addWidget(self.result_text)
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_acc)
        layout.addWidget(self.res_acc)
        self.res_acc.setReadOnly(True)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.plot_button)
        layout.addWidget(self.Measure_Acc)
        layout.addWidget(self.exit_button)

        self.decode_method_dropdown.currentIndexChanged.connect(
            self.clear_result_text)

        self.upload_button.clicked.connect(self.upload_audio_file)
        self.run_button.clicked.connect(self.run_decoder)
        self.plot_button.clicked.connect(self.plot_signal)

        self.exit_button.clicked.connect(self.exit_application)

        self.decoding_tab.setLayout(layout)

    def clear_result_text(self):
        # Clear the result text field
        self.result_text.clear()
        self.res_acc.clear()

    def encode_string(self):
        # Encode the input string
        input_str = self.input_text.text()

        # Check if the input is a string and contains only alphabetic characters
        if not isinstance(input_str, str) or not input_str.replace(" ", "").isalpha():
            # Display a message to the user in the GUI
            QMessageBox.warning(
                self, 'Invalid Input', 'Input must be a string containing only alphabetic characters.')
            return
        # Store the original string
        self.original_string = input_str
        encoded_signal = self.encoder.encode_string(
            input_str.lower())  # Convert to lowercase before encoding
        self.encoded_signal = encoded_signal

    def play_signal(self):
        # Play the encoded signal

        if hasattr(self, 'encoded_signal'):
            sd.play(self.encoded_signal, samplerate=8000)
            sd.wait()

    def save_signal(self):
        # Save the encoded signal to a WAV file
        if hasattr(self, 'encoded_signal'):
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(
                self, 'Save Signal', '', 'WAV Files (*.wav);;All Files (*)')
            if file_path:
                write(file_path, 8000, self.encoded_signal)

    def view_signal(self):
        # Open a SignalViewer to visualize the encoded signal
        if hasattr(self, 'encoded_signal'):
            viewer = SignalViewer(self.encoded_signal)
            viewer.exec_()

    def plot_dft(self):
        # Plot the DFT of the encoded signal
        if hasattr(self, 'encoded_signal'):
            freqs, dft = self.encoder.generate_dft(self.encoded_signal)
            plt.figure()
            plt.plot(freqs, dft)
            plt.title('DFT of Encoded Signal')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.xlim(-5000, 5000)

            for freq in self.encoder.character_frequencies['a']:
                plt.annotate(f'{freq} Hz', xy=(freq, max(dft)), xytext=(freq, max(dft)*1.1),
                             arrowprops=dict(facecolor='black', shrink=0.05))

            plt.show()

    def upload_audio_file(self):
        # Open a file dialog to upload an audio file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 'Upload Audio File', '', 'WAV Files (*.wav);;All Files (*)')
        self.audio_file_path = file_path

    def run_decoder(self):


        if hasattr(self, 'audio_file_path'):
            sample_rate, signal = read(self.audio_file_path)

            if self.decode_method_dropdown.currentText() == 'Bandpass Filters':
                self.decoded_result = self.decoder.decode_bandpass_filters(signal)
                length_of_encoded_string = len(self.decoded_result)  # Length
                correctness = 0
                for i in range(length_of_encoded_string):
                    if self.original_string[i] == self.decoded_result[i]:
                       correctness += 1
                accuracy = (int(correctness) / int(length_of_encoded_string)) * (100)
                self.result_text.setText(
                  f"Decoder Result:\n{self.decoded_result}")
                formatted_accuracy = f"\n{int(accuracy):.2f}%"
                self.res_acc.setText(formatted_accuracy)

            elif self.decode_method_dropdown.currentText() == 'Frequency Analysis (DFT)':
                self.decoded_result = self.decoder.decode_frequency_analysis(
                    signal)
                length_of_encoded_string = len(self.decoded_result)  # Length
                correctness = 0
                for i in range(length_of_encoded_string):
                    if self.original_string[i] == self.decoded_result[i]:
                        correctness += 1
                accuracy = (int(correctness) / int(length_of_encoded_string)) * (100)
                self.result_text.setText(
                    f"Decoder Result:\n{self.decoded_result}")
                formatted_accuracy = f"\n{int(accuracy):.2f}%"
                self.res_acc.setText(formatted_accuracy)
                
            else:
                self.decoded_result = 'Invalid decoding method'

            # Update the result text



    def plot_signal(self):
        # Plot the time-domain signal of the uploaded audio file
        if hasattr(self, 'audio_file_path'):
            sample_rate, signal = read(self.audio_file_path)

            time_axis = np.arange(len(signal)) / sample_rate
            sample_axis = np.arange(len(signal))

            plt.figure()
            plt.plot(sample_axis, signal)
            plt.title('Audio Signal')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.show()

    def exit_application(self):
        # Exit the application
        QApplication.quit()


if __name__ == '__main__':
    # Create and run the application
    app = QApplication([])
    window = EncoderDecoderGUI()
    window.show()
    app.exec_()
