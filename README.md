# Audio Processing Tools

Welcome to my GitHub profile! I'm sharing advanced audio processing tools for high-quality sound manipulation. These C-based utilities provide professional-grade audio enhancements for mastering and stereo processing.

## Tools Overview

### Enhanced Filter Bank

A sophisticated spectral smoothing processor with improved low-frequency handling and phase preservation.

**Features:**
- Frequency-dependent smoothing with precision control
- Low-end preservation for maintaining bass integrity
- Harsh frequency reduction (2-5kHz) for less fatiguing sound
- Subtle presence boost (7-10kHz) for clarity
- Optimized overlap-add processing with 87.5% window overlap

**Compilation:**
```bash
gcc -o enhanced_filter_bank enhanced_filter_bank.c -lsndfile -lm -lfftw3f
```

**Usage:**
```bash
./enhanced_filter_bank input.wav output.wav [options]
```

**Options:**
- `-bands <num>`: Number of frequency bands (default: 1024)
- `-smooth <amount>`: Smoothing amount (0.0-1.0, default: 0.6)
- `-preserve <amount>`: Low-end preservation (0.0-1.0, default: 0.8)
- `-harsh <db>`: Harsh frequencies reduction (0-12dB, default: 3.0)
- `-debug`: Enable debug mode

### K-Stereo Processor

A K-Stereo inspired processor for ambient recovery and natural stereo enhancement based on Bob Katz's mastering approach.

**Features:**
- Ambience recovery through sophisticated time-frequency analysis
- Depth enhancement using psychoacoustic cues
- Natural stereo widening without phase issues
- Mid-side or L/R processing modes
- Phase-coherent processing to avoid comb filtering
- Enhanced ambience extraction with coherence control

**Compilation:**
```bash
gcc -o k_stereo k_stereo.c -lsndfile -lm -lfftw3f
```

**Usage:**
```bash
./k_stereo input.wav output.wav [options]
```

**Options:**
- `-crossfeed <amount>`: Crossfeed amount (0.0-1.0, default: 0.3)
- `-ambience <amount>`: Ambience recovery amount (0.0-1.0, default: 0.6)
- `-depth <amount>`: Depth enhancement (0.0-1.0, default: 0.5)
- `-width <amount>`: Width enhancement (0.0-1.0, default: 0.5)
- `-lr`: Use L/R processing instead of M/S
- `-debug`: Enable debug mode
- `-basic`: Use basic delay-based ambience extraction
- `-enhanced`: Use enhanced PCA-based ambience extraction
- `-decorr <amount>`: Decorrelation amount (0.0-1.0, default: 0.7)
- `-coherence <thresh>`: Coherence threshold (0.0-1.0, default: 0.6)

## Dependencies

Both tools require:
- libsndfile (audio file I/O)
- FFTW3 (Fast Fourier Transform library)
- Standard C math libraries

### Installation on Ubuntu/Debian:
```bash
sudo apt-get install libsndfile1-dev libfftw3-dev
```

### Installation on macOS (via Homebrew):
```bash
brew install libsndfile fftw
```

## Applications

These tools are designed for:
- Music production and mastering
- Audio restoration
- Podcast and voice enhancement
- Film and game audio processing

## Technical Background

Both tools utilize spectral processing techniques with carefully tuned psychoacoustic parameters. The enhanced filter bank focuses on frequency-dependent smoothing with low-end preservation, while the K-Stereo processor enhances spatial perception through sophisticated ambience recovery.

The implementations include:
- Overlap-add FFT processing with customized window functions
- Multi-rate processing for optimal time-frequency resolution
- Phase-aware processing to maintain stereo imaging
- Perceptual weighting based on psychoacoustic principles

## License

[MIT License](LICENSE)

## Contact

Feel free to reach out with questions or feedback!

---

*Note: These tools are inspired by professional audio processing techniques but are not affiliated with any commercial products.*
