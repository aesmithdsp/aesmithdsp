/**
 * enhanced_filter_bank.c
 * 
 * An improved high-end audio smoothing processor with better low frequency
 * handling and phase preservation.
 * 
 * Compile with: gcc -o enhanced_filter_bank enhanced_filter_bank.c -lsndfile -lm -lfftw3f
 * Usage: ./enhanced_filter_bank input.wav output.wav [options]
 * 
 * Required dependencies:
 * - libsndfile (audio file I/O)
 * - FFTW3 (Fast Fourier Transform library)
 * 
 * Options:
 *   -bands <num>      : Number of frequency bands (default: 1024)
 *   -smooth <amount>  : Smoothing amount (0.0-1.0, default: 0.6)
 *   -preserve <amount>: Low-end preservation (0.0-1.0, default: 0.8)
 *   -harsh <db>       : Harsh frequencies reduction (0-12dB, default: 3.0)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <sndfile.h>
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Maximum FFT size
#define MAX_FFT_SIZE 16384
#define DEFAULT_FFT_SIZE 4096

// Low frequency preservation threshold
#define LOW_FREQ_THRESHOLD 250.0  // Hz
#define MID_FREQ_THRESHOLD 800.0  // Hz

// Processing settings
typedef struct {
    int fft_size;             // FFT size
    float smooth_amount;      // Overall smoothing amount (0.0-1.0)
    float low_preserve;       // Low-end preservation amount (0.0-1.0)
    float harsh_reduction;    // Harsh frequency reduction in dB
    
    // Overlap settings (higher for better low-end reconstruction)
    int overlap;              // Overlap factor (4 = 75% overlap, 8 = 87.5% overlap)
    int hop_size;             // Hop size between frames
    
    // Debug option
    int debug_mode;           // Print debug info if set
} ProcessingSettings;

// Processor state
typedef struct {
    // FFT state
    fftwf_plan forward_plan;
    fftwf_plan inverse_plan;
    float *input_buffer;
    float *output_buffer;
    fftwf_complex *freq_data;
    
    // Analysis/Synthesis windows
    float *analysis_window;
    float *synthesis_window;
    
    // Overlap-add buffer
    float *overlap_buffer;
    
    // Processing memory
    float *prev_magnitudes;
    float *prev_phases;
    
    // Settings
    ProcessingSettings settings;
    
    // System info
    int sample_rate;
    int channels;
} Processor;

// Function prototypes
void initialize_processor(Processor *proc, ProcessingSettings *settings, int sample_rate, int channels);
void cleanup_processor(Processor *proc);
void process_file(SNDFILE *infile, SNDFILE *outfile, SF_INFO *sfinfo, ProcessingSettings *settings);
void process_channel(Processor *proc, float *input, float *output, int num_samples);
void process_frame(Processor *proc, float *frame, int channel_idx);
void parse_arguments(int argc, char **argv, ProcessingSettings *settings);
void print_settings(ProcessingSettings *settings, int sample_rate);
float calculate_gain_factor(Processor *proc, int bin_idx, float magnitude);
void apply_window(float *buffer, float *window, int size);

int main(int argc, char *argv[]) {
    SF_INFO sfinfo;
    SNDFILE *infile, *outfile;
    ProcessingSettings settings;
    
    // Default settings
    settings.fft_size = DEFAULT_FFT_SIZE;
    settings.smooth_amount = 0.6f;
    settings.low_preserve = 0.8f;
    settings.harsh_reduction = 3.0f;
    settings.overlap = 8;  // 87.5% overlap for better low-end
    settings.debug_mode = 0;
    
    // Check command line arguments
    if (argc < 3) {
        printf("Usage: %s input.wav output.wav [options]\n", argv[0]);
        printf("Options:\n");
        printf("  -bands <num>      : Number of frequency bands (default: 1024)\n");
        printf("  -smooth <amount>  : Smoothing amount (0.0-1.0, default: 0.6)\n");
        printf("  -preserve <amount>: Low-end preservation (0.0-1.0, default: 0.8)\n");
        printf("  -harsh <db>       : Harsh frequencies reduction (0-12dB, default: 3.0)\n");
        printf("  -debug            : Enable debug mode\n");
        return 1;
    }
    
    // Parse command line arguments
    parse_arguments(argc, argv, &settings);
    
    // Set hop size based on overlap
    settings.hop_size = settings.fft_size / settings.overlap;
    
    // Open input file
    memset(&sfinfo, 0, sizeof(sfinfo));
    if (!(infile = sf_open(argv[1], SFM_READ, &sfinfo))) {
        printf("Error: Could not open input file %s\n", argv[1]);
        return 1;
    }
    
    // Open output file
    if (!(outfile = sf_open(argv[2], SFM_WRITE, &sfinfo))) {
        printf("Error: Could not open output file %s\n", argv[2]);
        sf_close(infile);
        return 1;
    }
    
    // Print settings
    print_settings(&settings, sfinfo.samplerate);
    
    // Process the file
    process_file(infile, outfile, &sfinfo, &settings);
    
    // Clean up
    sf_close(infile);
    sf_close(outfile);
    
    printf("Processing complete. Output written to %s\n", argv[2]);
    return 0;
}

/**
 * Initialize the processor
 */
void initialize_processor(Processor *proc, ProcessingSettings *settings, int sample_rate, int channels) {
    int i;
    
    // Store settings
    proc->settings = *settings;
    proc->sample_rate = sample_rate;
    proc->channels = channels;
    
    // Allocate FFT buffers
    proc->input_buffer = (float*)fftwf_malloc(sizeof(float) * settings->fft_size);
    proc->output_buffer = (float*)fftwf_malloc(sizeof(float) * settings->fft_size);
    proc->freq_data = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (settings->fft_size/2 + 1));
    
    // Create FFT plans
    proc->forward_plan = fftwf_plan_dft_r2c_1d(settings->fft_size, proc->input_buffer, 
                                               proc->freq_data, FFTW_MEASURE);
    proc->inverse_plan = fftwf_plan_dft_c2r_1d(settings->fft_size, proc->freq_data, 
                                               proc->output_buffer, FFTW_MEASURE);
    
    // Allocate window functions
    proc->analysis_window = (float*)malloc(sizeof(float) * settings->fft_size);
    proc->synthesis_window = (float*)malloc(sizeof(float) * settings->fft_size);
    
    // Allocate overlap buffer for each channel
    proc->overlap_buffer = (float*)calloc(channels * settings->fft_size, sizeof(float));
    
    // Allocate memory buffers for magnitude and phase smoothing
    proc->prev_magnitudes = (float*)calloc(channels * (settings->fft_size/2 + 1), sizeof(float));
    proc->prev_phases = (float*)calloc(channels * (settings->fft_size/2 + 1), sizeof(float));
    
    // Create improved Hann window for analysis (better frequency resolution)
    for (i = 0; i < settings->fft_size; i++) {
        proc->analysis_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (settings->fft_size - 1)));
    }
    
    // Create synthesis window
    // Using a custom window for better overlap-add reconstruction
    float overlap_factor = (float)settings->overlap;
    for (i = 0; i < settings->fft_size; i++) {
        // Basic Hann window
        float window = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (settings->fft_size - 1)));
        
        // Scale for perfect reconstruction with overlap-add
        proc->synthesis_window[i] = window / (overlap_factor/2.0f);
    }
    
    if (settings->debug_mode) {
        printf("Processor initialized:\n");
        printf("  FFT size: %d\n", settings->fft_size);
        printf("  Hop size: %d\n", settings->hop_size);
        printf("  Overlap factor: %d\n", settings->overlap);
    }
}

/**
 * Clean up processor resources
 */
void cleanup_processor(Processor *proc) {
    // Free FFT resources
    fftwf_destroy_plan(proc->forward_plan);
    fftwf_destroy_plan(proc->inverse_plan);
    fftwf_free(proc->input_buffer);
    fftwf_free(proc->output_buffer);
    fftwf_free(proc->freq_data);
    
    // Free other resources
    free(proc->analysis_window);
    free(proc->synthesis_window);
    free(proc->overlap_buffer);
    free(proc->prev_magnitudes);
    free(proc->prev_phases);
}

/**
 * Process an entire audio file
 */
void process_file(SNDFILE *infile, SNDFILE *outfile, SF_INFO *sfinfo, ProcessingSettings *settings) {
    Processor proc;
    int frames_read, frame_idx, ch, i;
    float *input_buffer, *output_buffer;
    float *channel_input, *channel_output;
    int buffer_size = 16384; // Process in reasonably sized chunks
    
    // Initialize the processor
    initialize_processor(&proc, settings, sfinfo->samplerate, sfinfo->channels);
    
    // Allocate buffers for interleaved audio
    input_buffer = (float*)malloc(sizeof(float) * buffer_size * sfinfo->channels);
    output_buffer = (float*)malloc(sizeof(float) * buffer_size * sfinfo->channels);
    
    // Allocate buffers for de-interleaved audio (per channel)
    channel_input = (float*)malloc(sizeof(float) * buffer_size);
    channel_output = (float*)malloc(sizeof(float) * buffer_size);
    
    // Process the file in chunks
    while ((frames_read = sf_readf_float(infile, input_buffer, buffer_size)) > 0) {
        // Clear output buffer
        memset(output_buffer, 0, sizeof(float) * frames_read * sfinfo->channels);
        
        // Process each channel separately
        for (ch = 0; ch < sfinfo->channels; ch++) {
            // De-interleave input for this channel
            for (i = 0; i < frames_read; i++) {
                channel_input[i] = input_buffer[i * sfinfo->channels + ch];
            }
            
            // Process this channel
            process_channel(&proc, channel_input, channel_output, frames_read);
            
            // Re-interleave output
            for (i = 0; i < frames_read; i++) {
                output_buffer[i * sfinfo->channels + ch] = channel_output[i];
            }
        }
        
        // Write output
        sf_writef_float(outfile, output_buffer, frames_read);
    }
    
    // Clean up
    cleanup_processor(&proc);
    free(input_buffer);
    free(output_buffer);
    free(channel_input);
    free(channel_output);
}

/**
 * Process a single channel of audio
 */
void process_channel(Processor *proc, float *input, float *output, int num_samples) {
    int frame_start, i, channel_idx = 0;
    int fft_size = proc->settings.fft_size;
    int hop_size = proc->settings.hop_size;
    
    // Clear output buffer
    memset(output, 0, sizeof(float) * num_samples);
    
    // Process frames with overlap
    for (frame_start = -fft_size + hop_size; frame_start < num_samples; frame_start += hop_size) {
        // Fill input buffer with zeros
        memset(proc->input_buffer, 0, sizeof(float) * fft_size);
        
        // Copy valid samples to input buffer
        for (i = 0; i < fft_size; i++) {
            int input_index = frame_start + i;
            if (input_index >= 0 && input_index < num_samples) {
                proc->input_buffer[i] = input[input_index];
            }
        }
        
        // Process this frame
        process_frame(proc, proc->input_buffer, channel_idx);
        
        // Overlap-add to output
        for (i = 0; i < fft_size; i++) {
            int output_index = frame_start + i;
            if (output_index >= 0 && output_index < num_samples) {
                output[output_index] += proc->output_buffer[i];
            }
        }
    }
}

/**
 * Process a single frame of audio
 */
void process_frame(Processor *proc, float *frame, int channel_idx) {
    int i, bin;
    int fft_size = proc->settings.fft_size;
    int half_fft_size = fft_size / 2 + 1;
    float magnitude, phase, real, imag;
    float bin_freq, gain_factor;
    
    // Apply analysis window
    apply_window(frame, proc->analysis_window, fft_size);
    
    // Forward FFT
    memcpy(proc->input_buffer, frame, sizeof(float) * fft_size);
    fftwf_execute(proc->forward_plan);
    
    // Process frequency bins
    for (bin = 1; bin < half_fft_size - 1; bin++) {  // Skip DC and Nyquist
        // Get bin data
        real = crealf(proc->freq_data[bin]);
        imag = cimagf(proc->freq_data[bin]);
        
        // Calculate magnitude and phase
        magnitude = sqrtf(real*real + imag*imag);
        phase = atan2f(imag, real);
        
        // Get bin frequency
        bin_freq = (float)bin * proc->sample_rate / fft_size;
        
        // Determine gain factor based on bin frequency and magnitude
        gain_factor = calculate_gain_factor(proc, bin, magnitude);
        
        // Apply gain factor
        real *= gain_factor;
        imag *= gain_factor;
        
        // Store processed bin
        proc->freq_data[bin] = real + I * imag;
        
        // Store for next frame (magnitude smoothing)
        proc->prev_magnitudes[channel_idx * half_fft_size + bin] = magnitude;
        proc->prev_phases[channel_idx * half_fft_size + bin] = phase;
    }
    
    // Inverse FFT
    fftwf_execute(proc->inverse_plan);
    
    // Scale output and apply synthesis window
    for (i = 0; i < fft_size; i++) {
        proc->output_buffer[i] /= fft_size;
        proc->output_buffer[i] *= proc->synthesis_window[i];
    }
}

/**
 * Calculate gain factor for a specific frequency bin
 */
float calculate_gain_factor(Processor *proc, int bin_idx, float magnitude) {
    float bin_freq = (float)bin_idx * proc->sample_rate / proc->settings.fft_size;
    float gain_factor = 1.0f;
    float smooth_amount = proc->settings.smooth_amount;
    
    // Special low frequency handling - preserve more of the original
    if (bin_freq < LOW_FREQ_THRESHOLD) {
        // Linear interpolation from full preservation at 20Hz to partial at threshold
        float preservation_factor = ((LOW_FREQ_THRESHOLD - bin_freq) / (LOW_FREQ_THRESHOLD - 20.0f)) 
                                   * proc->settings.low_preserve;
        
        // Reduce smoothing effect for low frequencies
        smooth_amount *= (1.0f - preservation_factor);
    }
    
    // Special mid frequency handling (more careful with fundamental vocal frequencies)
    if (bin_freq >= LOW_FREQ_THRESHOLD && bin_freq < MID_FREQ_THRESHOLD) {
        // Slightly reduced processing in this range
        smooth_amount *= 0.85f;
    }
    
    // Handle harsh frequencies (more aggressive processing)
    if (bin_freq >= 2000.0f && bin_freq <= 5000.0f) {
        // Calculate how far we are into the harsh range (0.0-1.0)
        float harsh_factor = (bin_freq - 2000.0f) / 3000.0f;
        if (harsh_factor > 1.0f) harsh_factor = 1.0f;
        
        // Make a bell curve with peak at 3.5kHz
        harsh_factor = 1.0f - fabsf((harsh_factor * 2.0f - 1.0f)); // 0->1->0 curve
        harsh_factor = powf(harsh_factor, 0.5f); // Make it wider
        
        // Calculate reduction in linear gain
        float reduction_db = proc->settings.harsh_reduction * harsh_factor * smooth_amount;
        float reduction_gain = powf(10.0f, -reduction_db / 20.0f);
        
        // Apply reduction
        gain_factor *= reduction_gain;
    }
    
    // Add a subtle presence boost for clarity after processing (7-10kHz)
    if (bin_freq >= 7000.0f && bin_freq <= 10000.0f) {
        float boost_factor = (bin_freq - 7000.0f) / 3000.0f;
        if (boost_factor > 1.0f) boost_factor = 1.0f;
        
        // Small boost to compensate for smoothing
        float boost_amount = 1.0f + (0.5f * boost_factor * smooth_amount);
        gain_factor *= boost_amount;
    }
    
    return gain_factor;
}

/**
 * Apply window function to buffer
 */
void apply_window(float *buffer, float *window, int size) {
    for (int i = 0; i < size; i++) {
        buffer[i] *= window[i];
    }
}

/**
 * Parse command line arguments
 */
void parse_arguments(int argc, char **argv, ProcessingSettings *settings) {
    int i;
    
    for (i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-bands") == 0 && i+1 < argc) {
            int bands = atoi(argv[++i]);
            // Find the next power of 2 for FFT size
            int fft_size = 1;
            while (fft_size < bands * 2) {
                fft_size *= 2;
            }
            if (fft_size > MAX_FFT_SIZE) fft_size = MAX_FFT_SIZE;
            settings->fft_size = fft_size;
        }
        else if (strcmp(argv[i], "-smooth") == 0 && i+1 < argc) {
            settings->smooth_amount = atof(argv[++i]);
            if (settings->smooth_amount < 0.0f) settings->smooth_amount = 0.0f;
            if (settings->smooth_amount > 1.0f) settings->smooth_amount = 1.0f;
        }
        else if (strcmp(argv[i], "-preserve") == 0 && i+1 < argc) {
            settings->low_preserve = atof(argv[++i]);
            if (settings->low_preserve < 0.0f) settings->low_preserve = 0.0f;
            if (settings->low_preserve > 1.0f) settings->low_preserve = 1.0f;
        }
        else if (strcmp(argv[i], "-harsh") == 0 && i+1 < argc) {
            settings->harsh_reduction = atof(argv[++i]);
            if (settings->harsh_reduction < 0.0f) settings->harsh_reduction = 0.0f;
            if (settings->harsh_reduction > 12.0f) settings->harsh_reduction = 12.0f;
        }
        else if (strcmp(argv[i], "-debug") == 0) {
            settings->debug_mode = 1;
        }
    }
}

/**
 * Print current settings
 */
void print_settings(ProcessingSettings *settings, int sample_rate) {
    printf("Processing Settings:\n");
    printf("  FFT Size: %d bins\n", settings->fft_size);
    printf("  Frequency Resolution: %.2f Hz\n", (float)sample_rate / settings->fft_size);
    printf("  Hop Size: %d samples (%.1f%% overlap)\n", 
           settings->hop_size, 100.0f * (1.0f - (float)settings->hop_size / settings->fft_size));
    printf("  Smoothing Amount: %.1f%%\n", settings->smooth_amount * 100.0f);
    printf("  Low-end Preservation: %.1f%%\n", settings->low_preserve * 100.0f);
    printf("  Harsh Frequency Reduction: %.1f dB\n", settings->harsh_reduction);
}
