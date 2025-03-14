/**
 * k_stereo.c
 * 
 * A K-Stereo inspired processor for ambient recovery and natural stereo enhancement
 * based on Bob Katz's approach to mastering-grade stereo processing.
 * 
 * Compile with: gcc -o k_stereo k_stereo.c -lsndfile -lm -lfftw3f
 * Usage: ./k_stereo input.wav output.wav [options]
 * 
 * Features:
 * - Ambience recovery through sophisticated time-frequency analysis
 * - Depth enhancement using psychoacoustic cues
 * - Natural stereo widening 
 * - Phase-coherent processing to avoid comb filtering
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <sndfile.h>
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Default settings
#define DEFAULT_FFT_SIZE 8192
#define DEFAULT_CROSSFEED 0.3f
#define DEFAULT_AMBIENCE 0.6f
#define DEFAULT_DEPTH 0.5f
#define DEFAULT_WIDTH 0.5f
#define DEFAULT_DECORR 0.7f
#define DEFAULT_COHERENCE 0.6f

// Minimum energy threshold to prevent processing silence
#define MIN_ENERGY_THRESHOLD 1e-8f

// Number of frames to fade in/out
#define FADE_FRAMES 4

// Processing settings
typedef struct {
    int fft_size;           // FFT size for spectral processing
    float crossfeed;        // Crossfeed amount (0.0-1.0)
    float ambience;         // Ambience recovery amount (0.0-1.0)
    float depth;            // Depth enhancement (0.0-1.0)
    float width;            // Width enhancement (0.0-1.0)
    int min_delay;          // Minimum delay for ambient extraction (samples)
    int max_delay;          // Maximum delay for ambient extraction (samples)
    int num_taps;           // Number of delay taps for ambient extraction
    int use_ms;             // Use mid-side processing instead of L/R
    int debug_mode;         // Enable debug output
    
    // Enhanced ambience parameters
    int use_enhanced_ambience;     // Whether to use the enhanced ambience algorithm
    float ambience_decorr;         // Amount of decorrelation in the ambience (0.0-1.0)
    float coherence_threshold;     // Threshold for determining coherent vs ambient
} KStereoSettings;

// Time-domain delays for ambience extraction
typedef struct {
    float *delay_buffers_l;  // Left channel delay buffers
    float *delay_buffers_r;  // Right channel delay buffers
    int *delay_lengths;      // Delay lengths
    float *delay_gains;      // Delay gains
    int buffer_pos;          // Current buffer position
    int num_delays;          // Number of delay lines
} DelayNetwork;

// FFT frame history for smooth transitions
typedef struct {
    fftwf_complex *prev_frame_freq_l;  // Previous frame data for left channel
    fftwf_complex *prev_frame_freq_r;  // Previous frame data for right channel
    float *prev_ambient_l;             // Previous ambient output for left channel
    float *prev_ambient_r;             // Previous ambient output for right channel
    int current_frame;                 // Frame counter for fade in/out
    bool initialized;                  // Whether history has been initialized
} FrameHistory;

// Spectral processing state
typedef struct {
    fftwf_plan forward_plan_l;
    fftwf_plan inverse_plan_l;
    fftwf_plan forward_plan_r;
    fftwf_plan inverse_plan_r;
    float *input_buffer_l;
    float *output_buffer_l;
    float *input_buffer_r;
    float *output_buffer_r;
    fftwf_complex *freq_data_l;
    fftwf_complex *freq_data_r;
    
    // Analysis windows
    float *analysis_window;
    float *synthesis_window;
    
    // Overlap-add buffers
    float *overlap_buffer_l;
    float *overlap_buffer_r;
    
    // Phase and correlation analysis
    float *phase_correlation;
    float *freq_width_factors;
    float *prev_coherence;      // Per-bin coherence history
    
    // Enhanced ambience extraction buffers
    float *temp_input_l;         // Pre-allocated FFT buffer
    float *temp_input_r;
    float *temp_output_l;
    float *temp_output_r;
    fftwf_complex *temp_freq_l;
    fftwf_complex *temp_freq_r;
    fftwf_plan fft_forward_l;    // Pre-allocated FFT plans
    fftwf_plan fft_forward_r;
    fftwf_plan fft_inverse_l;
    fftwf_plan fft_inverse_r;
    
    // Settings
    KStereoSettings settings;
    
    // Delay network for ambience extraction
    DelayNetwork delay_network;
    
    // Frame history for click elimination
    FrameHistory frame_history;
    
    // System info
    int sample_rate;
    int hop_size;
} KStereoProcessor;

// Function prototypes
void initialize_processor(KStereoProcessor *proc, KStereoSettings *settings, int sample_rate);
void cleanup_processor(KStereoProcessor *proc);
void reset_ambience_extraction_state(KStereoProcessor *proc);
void process_file(SNDFILE *infile, SNDFILE *outfile, SF_INFO *sfinfo, KStereoSettings *settings);
void process_frame(KStereoProcessor *proc, float *input_l, float *input_r, float *output_l, float *output_r, int num_samples);
void init_delay_network(KStereoProcessor *proc);
void process_delay_network(KStereoProcessor *proc, float in_l, float in_r, float *out_l, float *out_r);
void analyze_stereo_correlation(KStereoProcessor *proc);
void enhance_ambience(KStereoProcessor *proc);
void extract_enhanced_ambience(KStereoProcessor *proc, float *input_l, float *input_r, float *ambient_l, float *ambient_r, int num_samples);
void apply_window(float *buffer, float *window, int size);
void parse_arguments(int argc, char **argv, KStereoSettings *settings);
void print_settings(KStereoSettings *settings);
float calculate_coherence(float real_l, float imag_l, float real_r, float imag_r, float prev_coherence);

int main(int argc, char *argv[]) {
    SF_INFO sfinfo;
    SNDFILE *infile, *outfile;
    KStereoSettings settings;
    
    // Default settings
    settings.fft_size = DEFAULT_FFT_SIZE;
    settings.crossfeed = DEFAULT_CROSSFEED;
    settings.ambience = DEFAULT_AMBIENCE;
    settings.depth = DEFAULT_DEPTH;
    settings.width = DEFAULT_WIDTH;
    settings.min_delay = 15;       // 15 samples minimum delay (~ 0.3ms at 44.1kHz)
    settings.max_delay = 120;      // 120 samples maximum delay (~ 2.7ms at 44.1kHz)
    settings.num_taps = 8;         // 8 delay taps for ambience extraction
    settings.use_ms = 1;           // Use mid-side processing by default
    settings.debug_mode = 0;
    
    // New enhanced ambience parameters
    settings.use_enhanced_ambience = 0;  // Use enhanced ambience by default
    settings.ambience_decorr = DEFAULT_DECORR;
    settings.coherence_threshold = DEFAULT_COHERENCE;
    
    // Check command line arguments
    if (argc < 3) {
        printf("Usage: %s input.wav output.wav [options]\n", argv[0]);
        printf("Options:\n");
        printf("  -crossfeed <amount> : Crossfeed amount (0.0-1.0, default: %.1f)\n", DEFAULT_CROSSFEED);
        printf("  -ambience <amount>  : Ambience recovery amount (0.0-1.0, default: %.1f)\n", DEFAULT_AMBIENCE);
        printf("  -depth <amount>     : Depth enhancement (0.0-1.0, default: %.1f)\n", DEFAULT_DEPTH);
        printf("  -width <amount>     : Width enhancement (0.0-1.0, default: %.1f)\n", DEFAULT_WIDTH);
        printf("  -lr                 : Use L/R processing instead of M/S\n");
        printf("  -debug              : Enable debug mode\n");
        printf("  -basic              : Use basic delay-based ambience extraction\n");
        printf("  -enhanced           : Use enhanced PCA-based ambience extraction (experimental)\n");
        printf("  -decorr <amount>    : Decorrelation amount (0.0-1.0, default: %.1f)\n", DEFAULT_DECORR);
        printf("  -coherence <thresh> : Coherence threshold (0.0-1.0, default: %.1f)\n", DEFAULT_COHERENCE);
        return 1;
    }
    
    // Parse command line arguments
    parse_arguments(argc, argv, &settings);
    
    // Open input file
    memset(&sfinfo, 0, sizeof(sfinfo));
    if (!(infile = sf_open(argv[1], SFM_READ, &sfinfo))) {
        printf("Error: Could not open input file %s\n", argv[1]);
        return 1;
    }
    
    // Check if input is stereo
    if (sfinfo.channels != 2) {
        printf("Error: K-Stereo processing requires a stereo input file\n");
        sf_close(infile);
        return 1;
    }
    
    // Open output file
    if (!(outfile = sf_open(argv[2], SFM_WRITE, &sfinfo))) {
        printf("Error: Could not open output file %s\n", argv[2]);
        sf_close(infile);
        return 1;
    }
    
    // Print settings
    print_settings(&settings);
    
    // Process the file
    process_file(infile, outfile, &sfinfo, &settings);
    
    // Clean up
    sf_close(infile);
    sf_close(outfile);
    
    printf("Processing complete. Output written to %s\n", argv[2]);
    return 0;
}

/**
 * Initialize the K-Stereo processor
 */
void initialize_processor(KStereoProcessor *proc, KStereoSettings *settings, int sample_rate) {
    int i;
    int half_fft_size = settings->fft_size / 2 + 1;
    
    // Store settings
    proc->settings = *settings;
    proc->sample_rate = sample_rate;
    proc->hop_size = settings->fft_size / 4; // 75% overlap
    
    // Allocate FFT buffers with error checking
    proc->input_buffer_l = (float*)fftwf_malloc(sizeof(float) * settings->fft_size);
    if (!proc->input_buffer_l) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    proc->output_buffer_l = (float*)fftwf_malloc(sizeof(float) * settings->fft_size);
    if (!proc->output_buffer_l) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    proc->input_buffer_r = (float*)fftwf_malloc(sizeof(float) * settings->fft_size);
    if (!proc->input_buffer_r) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    proc->output_buffer_r = (float*)fftwf_malloc(sizeof(float) * settings->fft_size);
    if (!proc->output_buffer_r) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    proc->freq_data_l = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * half_fft_size);
    if (!proc->freq_data_l) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    proc->freq_data_r = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * half_fft_size);
    if (!proc->freq_data_r) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    // Create FFT plans
    proc->forward_plan_l = fftwf_plan_dft_r2c_1d(settings->fft_size, proc->input_buffer_l, 
                                                proc->freq_data_l, FFTW_MEASURE);
    proc->inverse_plan_l = fftwf_plan_dft_c2r_1d(settings->fft_size, proc->freq_data_l, 
                                                proc->output_buffer_l, FFTW_MEASURE);
    
    proc->forward_plan_r = fftwf_plan_dft_r2c_1d(settings->fft_size, proc->input_buffer_r, 
                                                proc->freq_data_r, FFTW_MEASURE);
    proc->inverse_plan_r = fftwf_plan_dft_c2r_1d(settings->fft_size, proc->freq_data_r, 
                                                proc->output_buffer_r, FFTW_MEASURE);
    
    // Allocate window functions
    proc->analysis_window = (float*)malloc(sizeof(float) * settings->fft_size);
    if (!proc->analysis_window) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    proc->synthesis_window = (float*)malloc(sizeof(float) * settings->fft_size);
    if (!proc->synthesis_window) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    // Allocate overlap buffers
    proc->overlap_buffer_l = (float*)calloc(settings->fft_size, sizeof(float));
    if (!proc->overlap_buffer_l) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    proc->overlap_buffer_r = (float*)calloc(settings->fft_size, sizeof(float));
    if (!proc->overlap_buffer_r) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    // Allocate correlation analysis buffers
    proc->phase_correlation = (float*)calloc(half_fft_size, sizeof(float));
    if (!proc->phase_correlation) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    proc->freq_width_factors = (float*)calloc(half_fft_size, sizeof(float));
    if (!proc->freq_width_factors) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    // Allocate per-bin coherence history
    proc->prev_coherence = (float*)calloc(half_fft_size, sizeof(float));
    if (!proc->prev_coherence) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    // Pre-allocate ambience extraction buffers
    proc->temp_input_l = (float*)fftwf_malloc(sizeof(float) * settings->fft_size);
    if (!proc->temp_input_l) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    proc->temp_input_r = (float*)fftwf_malloc(sizeof(float) * settings->fft_size);
    if (!proc->temp_input_r) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    proc->temp_output_l = (float*)fftwf_malloc(sizeof(float) * settings->fft_size);
    if (!proc->temp_output_l) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    proc->temp_output_r = (float*)fftwf_malloc(sizeof(float) * settings->fft_size);
    if (!proc->temp_output_r) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    proc->temp_freq_l = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * half_fft_size);
    if (!proc->temp_freq_l) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    proc->temp_freq_r = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * half_fft_size);
    if (!proc->temp_freq_r) {
        fprintf(stderr, "Memory allocation failed in initialize_processor\n");
        exit(1);
    }
    
    // Create FFT plans for ambience extraction
    proc->fft_forward_l = fftwf_plan_dft_r2c_1d(settings->fft_size, proc->temp_input_l, proc->temp_freq_l, FFTW_MEASURE);
    proc->fft_forward_r = fftwf_plan_dft_r2c_1d(settings->fft_size, proc->temp_input_r, proc->temp_freq_r, FFTW_MEASURE);
    proc->fft_inverse_l = fftwf_plan_dft_c2r_1d(settings->fft_size, proc->temp_freq_l, proc->temp_output_l, FFTW_MEASURE);
    proc->fft_inverse_r = fftwf_plan_dft_c2r_1d(settings->fft_size, proc->temp_freq_r, proc->temp_output_r, FFTW_MEASURE);
    
    // Create improved analysis window (Hann window)
    for (i = 0; i < settings->fft_size; i++) {
        proc->analysis_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (settings->fft_size - 1)));
    }
    
    // Create synthesis window for perfect reconstruction with 75% overlap
    for (i = 0; i < settings->fft_size; i++) {
        // Using same window as analysis
        float window = proc->analysis_window[i];
        // Scale for perfect reconstruction with overlap-add (4 windows overlapping)
        proc->synthesis_window[i] = window / 2.0f;
    }
    
    // Initialize delay network for ambience extraction
    init_delay_network(proc);
    
    // Initialize frame history
    proc->frame_history.prev_frame_freq_l = NULL;
    proc->frame_history.prev_frame_freq_r = NULL;
    proc->frame_history.prev_ambient_l = NULL;
    proc->frame_history.prev_ambient_r = NULL;
    proc->frame_history.current_frame = 0;
    proc->frame_history.initialized = false;
}

/**
 * Reset ambience extraction state between files
 */
void reset_ambience_extraction_state(KStereoProcessor *proc) {
    int half_fft_size = proc->settings.fft_size / 2 + 1;
    
    // Free previous history if it exists
    if (proc->frame_history.prev_frame_freq_l != NULL) {
        fftwf_free(proc->frame_history.prev_frame_freq_l);
        fftwf_free(proc->frame_history.prev_frame_freq_r);
        free(proc->frame_history.prev_ambient_l);
        free(proc->frame_history.prev_ambient_r);
    }
    
    // Allocate new history buffers
    proc->frame_history.prev_frame_freq_l = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * half_fft_size);
    if (!proc->frame_history.prev_frame_freq_l) {
        fprintf(stderr, "Memory allocation failed in reset_ambience_extraction_state\n");
        exit(1);
    }
    
    proc->frame_history.prev_frame_freq_r = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * half_fft_size);
    if (!proc->frame_history.prev_frame_freq_r) {
        fprintf(stderr, "Memory allocation failed in reset_ambience_extraction_state\n");
        exit(1);
    }
    
    proc->frame_history.prev_ambient_l = (float*)calloc(proc->settings.fft_size, sizeof(float));
    if (!proc->frame_history.prev_ambient_l) {
        fprintf(stderr, "Memory allocation failed in reset_ambience_extraction_state\n");
        exit(1);
    }
    
    proc->frame_history.prev_ambient_r = (float*)calloc(proc->settings.fft_size, sizeof(float));
    if (!proc->frame_history.prev_ambient_r) {
        fprintf(stderr, "Memory allocation failed in reset_ambience_extraction_state\n");
        exit(1);
    }
    
    // Clear history buffers
    memset(proc->frame_history.prev_frame_freq_l, 0, sizeof(fftwf_complex) * half_fft_size);
    memset(proc->frame_history.prev_frame_freq_r, 0, sizeof(fftwf_complex) * half_fft_size);
    memset(proc->frame_history.prev_ambient_l, 0, sizeof(float) * proc->settings.fft_size);
    memset(proc->frame_history.prev_ambient_r, 0, sizeof(float) * proc->settings.fft_size);
    
    // Reset coherence history - FIXED: initialize prev_coherence
    memset(proc->prev_coherence, 0, sizeof(float) * half_fft_size);
    
    proc->frame_history.current_frame = 0;
    proc->frame_history.initialized = true;
    
    // Reset delay network buffer position
    proc->delay_network.buffer_pos = 0;
    
    // Reset overlap buffers - FIXED: Reset overlap buffers for clean start
    memset(proc->overlap_buffer_l, 0, sizeof(float) * proc->settings.fft_size);
    memset(proc->overlap_buffer_r, 0, sizeof(float) * proc->settings.fft_size);
}

/**
 * Initialize the delay network for ambience extraction
 */
void init_delay_network(KStereoProcessor *proc) {
    int i;
    KStereoSettings *settings = &proc->settings;
    
    // Allocate delay network
    proc->delay_network.num_delays = settings->num_taps;
    proc->delay_network.delay_lengths = (int*)malloc(sizeof(int) * settings->num_taps);
    proc->delay_network.delay_gains = (float*)malloc(sizeof(float) * settings->num_taps);
    
    // Calculate maximum delay buffer needed
    int max_delay = settings->max_delay;
    int buffer_size = max_delay + 1;
    
    // Allocate delay buffers
    proc->delay_network.delay_buffers_l = (float*)calloc(buffer_size * settings->num_taps, sizeof(float));
    proc->delay_network.delay_buffers_r = (float*)calloc(buffer_size * settings->num_taps, sizeof(float));
    proc->delay_network.buffer_pos = 0;
    
    // Configure delay times and gains
    // The original K-Stereo likely used a more sophisticated approach with specific delay times
    // tuned by ear over years of development, but we'll use a simple approach here
    for (i = 0; i < settings->num_taps; i++) {
        // Distribute delays logarithmically between min and max
        float t = (float)i / (settings->num_taps - 1);
        float log_factor = powf(settings->max_delay / (float)settings->min_delay, t);
        int delay = (int)(settings->min_delay * log_factor);
        
        // Ensure delay is within bounds
        if (delay < settings->min_delay) delay = settings->min_delay;
        if (delay > settings->max_delay) delay = settings->max_delay;
        
        proc->delay_network.delay_lengths[i] = delay;
        
        // Assign gains - higher delays get lower gains
        // Again, the original likely used specific values tuned by ear
        proc->delay_network.delay_gains[i] = 1.0f - 0.7f * t;
    }
    
    if (settings->debug_mode) {
        printf("Delay network initialized with %d taps:\n", settings->num_taps);
        for (i = 0; i < settings->num_taps; i++) {
            printf("  Tap %d: %d samples (%.2fms) with gain %.3f\n", 
                   i, proc->delay_network.delay_lengths[i],
                   1000.0f * proc->delay_network.delay_lengths[i] / proc->sample_rate,
                   proc->delay_network.delay_gains[i]);
        }
    }
}

/**
 * Process a single frame through the delay network
 */
void process_delay_network(KStereoProcessor *proc, float in_l, float in_r, float *out_l, float *out_r) {
    int i, tap_idx, buffer_idx;
    DelayNetwork *network = &proc->delay_network;
    
    // Initialize outputs
    *out_l = 0.0f;
    *out_r = 0.0f;
    
    // Write inputs to current buffer position
    for (i = 0; i < network->num_delays; i++) {
        buffer_idx = i * (proc->settings.max_delay + 1) + network->buffer_pos;
        network->delay_buffers_l[buffer_idx] = in_l;
        network->delay_buffers_r[buffer_idx] = in_r;
    }
    
    // Read from delay lines and sum outputs
    for (i = 0; i < network->num_delays; i++) {
        // Calculate read position
        int read_pos = (network->buffer_pos - network->delay_lengths[i] + proc->settings.max_delay + 1) 
                      % (proc->settings.max_delay + 1);
        
        buffer_idx = i * (proc->settings.max_delay + 1) + read_pos;
        
        // Apply crossfeed - this creates lateral ambience similar to what K-Stereo did
        // Left channel gets right channel with delay and gain
        *out_l += network->delay_gains[i] * network->delay_buffers_r[buffer_idx] * proc->settings.crossfeed;
        
        // Right channel gets left channel with delay and gain
        *out_r += network->delay_gains[i] * network->delay_buffers_l[buffer_idx] * proc->settings.crossfeed;
    }
    
    // Update buffer position
    network->buffer_pos = (network->buffer_pos + 1) % (proc->settings.max_delay + 1);
    
    // Scale by ambience amount
    *out_l *= proc->settings.ambience;
    *out_r *= proc->settings.ambience;
}

/**
 * Calculate coherence between two signals in frequency domain
 */
float calculate_coherence(float real_l, float imag_l, float real_r, float imag_r, float prev_coherence) {
    float mag_l = sqrtf(real_l * real_l + imag_l * imag_l);
    float mag_r = sqrtf(real_r * real_r + imag_r * imag_r);
    
    // Skip calculation if magnitude is too small (avoid div by 0)
    if (mag_l < 1e-10f || mag_r < 1e-10f) {
        return prev_coherence; // Use previous value for stability
    }
    
    float phase_l = atan2f(imag_l, real_l);
    float phase_r = atan2f(imag_r, real_r);
    
    float phase_diff = phase_l - phase_r;
    // Wrap to -π to π
    while (phase_diff > M_PI) phase_diff -= 2.0f * M_PI;
    while (phase_diff < -M_PI) phase_diff += 2.0f * M_PI;
    
    // Calculate coherence with both phase and level differences
    float coherence = cosf(phase_diff) * (1.0f / (1.0f + fabsf(mag_l/mag_r - 1.0f) * 0.1f));
    
    // Apply temporal smoothing
    return 0.7f * coherence + 0.3f * prev_coherence;
}

/**
 * Enhanced ambience extraction using advanced decorrelation techniques
 */
void extract_enhanced_ambience(KStereoProcessor *proc, float *input_l, float *input_r, 
                               float *ambient_l, float *ambient_r, int num_samples) {
    // Declare variables
    int i, j, bin;
    int fft_size = proc->settings.fft_size;
    int half_fft_size = fft_size / 2 + 1;
    FrameHistory *history = &proc->frame_history;
    
    // Reset ambience extraction state is now handled only in process_file()
    
    // Clear ambient output buffers
    memset(ambient_l, 0, sizeof(float) * num_samples);
    memset(ambient_r, 0, sizeof(float) * num_samples);
    
    // Use pre-allocated buffers instead of allocating new ones
    float *temp_input_l = proc->temp_input_l;
    float *temp_input_r = proc->temp_input_r;
    float *temp_output_l = proc->temp_output_l;
    float *temp_output_r = proc->temp_output_r;
    fftwf_complex *temp_freq_l = proc->temp_freq_l;
    fftwf_complex *temp_freq_r = proc->temp_freq_r;
    
    // Process the signal in overlapping blocks for smoother results
    for (i = 0; i < num_samples; i += fft_size / 4) {
        // Clear the temporary buffers
        memset(temp_input_l, 0, sizeof(float) * fft_size);
        memset(temp_input_r, 0, sizeof(float) * fft_size);
        
        // Copy input samples into temporary buffers with windowing
        for (j = 0; j < fft_size && i + j < num_samples; j++) {
            // Apply a Hann window
            float window = 0.5f * (1.0f - cosf(2.0f * M_PI * j / (fft_size - 1)));
            temp_input_l[j] = input_l[i + j] * window;
            temp_input_r[j] = input_r[i + j] * window;
        }
        
        // Perform FFT using pre-allocated plans
        fftwf_execute(proc->fft_forward_l);
        fftwf_execute(proc->fft_forward_r);
        
        // FIXED: Better fade calculation for EOF
        float fade_factor = 1.0f;
        if (history->current_frame < FADE_FRAMES) {
            fade_factor = (float)history->current_frame / FADE_FRAMES;
        }
        if (i + fft_size >= num_samples) {
            float end_fade = (float)(num_samples - i) / (float)fft_size;
            fade_factor = fminf(fade_factor, end_fade);
        }
        
        // Apply frame transition smoothing for each bin
        for (bin = 0; bin < half_fft_size; bin++) {
            float real_l = crealf(temp_freq_l[bin]);
            float imag_l = cimagf(temp_freq_l[bin]);
            float real_r = crealf(temp_freq_r[bin]);
            float imag_r = cimagf(temp_freq_r[bin]);
            
            // Calculate bin energy
            float bin_energy = (real_l * real_l + imag_l * imag_l) + (real_r * real_r + imag_r * imag_r);
            
            // Skip processing very low level signals
            if (bin_energy < MIN_ENERGY_THRESHOLD) {
                temp_freq_l[bin] = 0;
                temp_freq_r[bin] = 0;
                continue;
            }
            
            // Apply temporal frame smoothing to prevent clicks
            if (history->current_frame > 0) {
                // Apply smoothing between frames to prevent discontinuities
                // Apply temporal smoothing between frames
                float smooth_factor = 0.7f;
                temp_freq_l[bin] = temp_freq_l[bin] * smooth_factor + 
                                   history->prev_frame_freq_l[bin] * (1.0f - smooth_factor);
                temp_freq_r[bin] = temp_freq_r[bin] * smooth_factor + 
                                   history->prev_frame_freq_r[bin] * (1.0f - smooth_factor);
                
                // Apply fade factor if needed
                if (fade_factor < 1.0f) {
                    temp_freq_l[bin] *= fade_factor;
                    temp_freq_r[bin] *= fade_factor;
                }
            }
            
            // Store for next frame
            history->prev_frame_freq_l[bin] = temp_freq_l[bin];
            history->prev_frame_freq_r[bin] = temp_freq_r[bin];
            
            // Get fresh values after smoothing
            real_l = crealf(temp_freq_l[bin]);
            imag_l = cimagf(temp_freq_l[bin]);
            real_r = crealf(temp_freq_r[bin]);
            imag_r = cimagf(temp_freq_r[bin]);
            
            // Calculate magnitudes and phases
            float mag_l = sqrtf(real_l * real_l + imag_l * imag_l);
            float mag_r = sqrtf(real_r * real_r + imag_r * imag_r);
            
            // Calculate coherence with temporal smoothing using stored per-bin coherence
            float coherence = calculate_coherence(real_l, imag_l, real_r, imag_r, proc->prev_coherence[bin]);
            proc->prev_coherence[bin] = coherence; // Update stored coherence
            
            // Separate into coherent and ambient components using PCA-inspired approach
            float coherent_weight = powf(fabsf(coherence), 2.0f); // Squared for sharper separation
            float ambient_weight = 1.0f - coherent_weight;
            
            // Apply user-defined coherence threshold
            if (coherent_weight > proc->settings.coherence_threshold) {
                coherent_weight = 1.0f;
                ambient_weight = 0.0f;
            }
            
            // Apply psychoacoustic weighting based on frequency - FIXED: use fminf for safety
            float bin_freq = (float)bin * proc->sample_rate / fft_size;
            if (bin_freq > 300.0f && bin_freq < 3000.0f) {
                // Enhance ambience in the midrange where spatial hearing is most sensitive
                ambient_weight = fminf(ambient_weight * 1.2f, 1.0f);
            }
            
            // Extract phase information for decorrelation
            float phase_l = atan2f(imag_l, real_l);
            float phase_r = atan2f(imag_r, real_r);
            
            // Create decorrelated version with frequency-dependent phase shift
            float decorr_amount = proc->settings.ambience_decorr;
            float decorr_phase_shift = ((float)bin / half_fft_size) * M_PI * decorr_amount;
            
            // Apply different decorrelation phase shifts to create natural ambience
            float decorr_real_l = mag_l * cosf(phase_l + decorr_phase_shift);
            float decorr_imag_l = mag_l * sinf(phase_l + decorr_phase_shift);
            float decorr_real_r = mag_r * cosf(phase_r - decorr_phase_shift);
            float decorr_imag_r = mag_r * sinf(phase_r - decorr_phase_shift);
            
            // FIXED: Better energy normalization with improved floating-point handling
            float orig_energy_l = sqrtf(real_l * real_l + imag_l * imag_l) * ambient_weight;
            float decorr_energy_l = sqrtf(decorr_real_l * decorr_real_l + decorr_imag_l * decorr_imag_l);
            float orig_energy_r = sqrtf(real_r * real_r + imag_r * imag_r) * ambient_weight;
            float decorr_energy_r = sqrtf(decorr_real_r * decorr_real_r + decorr_imag_r * decorr_imag_r);
            
            float scale_l = 0.7f;
            float scale_r = 0.7f;
            
            if (decorr_energy_l > 1e-8f) {
                scale_l = fminf(orig_energy_l / decorr_energy_l * 0.7f, 1.0f);
            }
            if (decorr_energy_r > 1e-8f) {
                scale_r = fminf(orig_energy_r / decorr_energy_r * 0.7f, 1.0f);
            }
            
            float ambient_real_l = real_l * ambient_weight + decorr_real_l * scale_l;
            float ambient_imag_l = imag_l * ambient_weight + decorr_imag_l * scale_l;
            float ambient_real_r = real_r * ambient_weight + decorr_real_r * scale_r;
            float ambient_imag_r = imag_r * ambient_weight + decorr_imag_r * scale_r;
            
            // Store ambient components only
            temp_freq_l[bin] = ambient_real_l + ambient_imag_l * I;
            temp_freq_r[bin] = ambient_real_r + ambient_imag_r * I;
        }
        
        // Apply inverse FFT to get time-domain ambient signal
        fftwf_execute(proc->fft_inverse_l);
        fftwf_execute(proc->fft_inverse_r);
        
        // Normalize and apply window for overlap-add
        for (j = 0; j < fft_size && i + j < num_samples; j++) {
            float window = 0.5f * (1.0f - cosf(2.0f * M_PI * j / (fft_size - 1)));
            float norm_l = temp_output_l[j] / fft_size * window;
            float norm_r = temp_output_r[j] / fft_size * window;
            
            // Add to ambient output with proper scaling
            if (i + j < num_samples) {
                // Apply cross-fade with previous frame output for smoother transitions
                if (history->current_frame > 0 && j < fft_size / 4) {
                    float fade = (float)j / (fft_size / 4);
                    int prev_idx = j + 3 * (fft_size / 4);
                    if (prev_idx >= 0 && prev_idx < fft_size) {
                        ambient_l[i + j] += norm_l * proc->settings.ambience * fade + 
                                            history->prev_ambient_l[prev_idx] * (1.0f - fade);
                        ambient_r[i + j] += norm_r * proc->settings.ambience * fade + 
                                            history->prev_ambient_r[prev_idx] * (1.0f - fade);
                    } else {
                        ambient_l[i + j] += norm_l * proc->settings.ambience;
                        ambient_r[i + j] += norm_r * proc->settings.ambience;
                    }
                } else {
                    ambient_l[i + j] += norm_l * proc->settings.ambience;
                    ambient_r[i + j] += norm_r * proc->settings.ambience;
                }
            }
        }
        
        // Store current frame output for next overlap
        for (j = 0; j < fft_size; j++) {
            history->prev_ambient_l[j] = temp_output_l[j] / fft_size * proc->settings.ambience;
            history->prev_ambient_r[j] = temp_output_r[j] / fft_size * proc->settings.ambience;
        }
        
        // Increment frame counter
        history->current_frame++;
    }
}

/**
 * Analyze stereo correlation in frequency domain
 */
void analyze_stereo_correlation(KStereoProcessor *proc) {
    int bin;
    int half_fft_size = proc->settings.fft_size / 2 + 1;
    
    // Analyze the correlation between left and right channels in frequency domain
    for (bin = 0; bin < half_fft_size; bin++) {
        float real_l = crealf(proc->freq_data_l[bin]);
        float imag_l = cimagf(proc->freq_data_l[bin]);
        float real_r = crealf(proc->freq_data_r[bin]);
        float imag_r = cimagf(proc->freq_data_r[bin]);
        
        // Calculate magnitudes
        float mag_l = sqrtf(real_l*real_l + imag_l*imag_l);
        float mag_r = sqrtf(real_r*real_r + imag_r*imag_r);
        
        // Calculate phase angles
        float phase_l = atan2f(imag_l, real_l);
        float phase_r = atan2f(imag_r, real_r);
        
        // Phase difference between channels
        float phase_diff = phase_l - phase_r;
        // Wrap to -π to π
        while (phase_diff > M_PI) phase_diff -= 2.0f * M_PI;
        while (phase_diff < -M_PI) phase_diff += 2.0f * M_PI;
        
        // Calculate correlation (1.0 = perfect correlation, -1.0 = anti-phase)
        float correlation = cosf(phase_diff);
        
        // Store correlation value
        proc->phase_correlation[bin] = correlation;
        
        // Calculate frequency-dependent width factor based on correlation
        // Lower correlation = more anticorrelated = more width
        // This is similar to how K-Stereo detected natural ambience
        proc->freq_width_factors[bin] = (1.0f - fabsf(correlation)) * proc->settings.width;
        
        // Boost width factor in mid-high frequencies (2-10kHz) to enhance spatial cues
        // This is based on psychoacoustic principles
        float bin_freq = (float)bin * proc->sample_rate / proc->settings.fft_size;
        if (bin_freq > 2000.0f && bin_freq < 10000.0f) {
            // Peak width enhancement at around 5-7kHz where human ears are most sensitive to spatial cues
            float freq_factor = 1.0f;
            if (bin_freq > 5000.0f && bin_freq < 7000.0f) {
                freq_factor = 1.5f;
            }
            proc->freq_width_factors[bin] *= freq_factor;
        }
        
        // Limit maximum width factor
        if (proc->freq_width_factors[bin] > 1.0f) {
            proc->freq_width_factors[bin] = 1.0f;
        }
    }
}

/**
 * Enhance ambience in frequency domain
 */
void enhance_ambience(KStereoProcessor *proc) {
    int bin;
    int half_fft_size = proc->settings.fft_size / 2 + 1;
    float width, depth_factor;
    
    // Apply frequency-dependent ambience enhancement
    for (bin = 1; bin < half_fft_size - 1; bin++) {
        float real_l = crealf(proc->freq_data_l[bin]);
        float imag_l = cimagf(proc->freq_data_l[bin]);
        float real_r = crealf(proc->freq_data_r[bin]);
        float imag_r = cimagf(proc->freq_data_r[bin]);
        
        if (proc->settings.use_ms) {
            // Convert to mid-side representation
            float real_mid = (real_l + real_r) * 0.5f;
            float imag_mid = (imag_l + imag_r) * 0.5f;
            float real_side = (real_l - real_r) * 0.5f;
            float imag_side = (imag_l - imag_r) * 0.5f;
            
            // Get bin frequency
            float bin_freq = (float)bin * proc->sample_rate / proc->settings.fft_size;
            
            // Calculate depth enhancement factor based on frequency
            // Enhance sense of depth by carefully processing mid frequencies
            if (bin_freq > 300.0f && bin_freq < 3000.0f) {
                depth_factor = proc->settings.depth;
            } else {
                depth_factor = proc->settings.depth * 0.5f;
            }
            
            // Get width factor for this bin
            width = proc->freq_width_factors[bin];
            
            // Apply width enhancement by boosting the side channel
            real_side *= (1.0f + width);
            imag_side *= (1.0f + width);
            
            // Apply depth enhancement by slightly attenuating mids in certain frequencies
            // and modifying the phase relationship
            if (bin_freq > 300.0f && bin_freq < 3000.0f) {
                real_mid *= (1.0f - 0.2f * depth_factor);
                imag_mid *= (1.0f - 0.2f * depth_factor);
            }
            
            // Convert back to L/R
            real_l = real_mid + real_side;
            imag_l = imag_mid + imag_side;
            real_r = real_mid - real_side;
            imag_r = imag_mid - imag_side;
            
            // Store processed data
            proc->freq_data_l[bin] = real_l + I * imag_l;
            proc->freq_data_r[bin] = real_r + I * imag_r;
        } else {
            // Direct L/R processing (not MS)
            // Less preferred for mastering but still an option
            
            // Calculate frequency-dependent processing
            float bin_freq = (float)bin * proc->sample_rate / proc->settings.fft_size;
            width = proc->freq_width_factors[bin];
            
            // Calculate magnitudes and phases
            float mag_l = sqrtf(real_l*real_l + imag_l*imag_l);
            float mag_r = sqrtf(real_r*real_r + imag_r*imag_r);
            float phase_l = atan2f(imag_l, real_l);
            float phase_r = atan2f(imag_r, real_r);
            
            // Enhance width through subtle phase and magnitude adjustments
            // The phase adjustments here are simplified compared to what K-Stereo likely did
            float phase_diff = phase_l - phase_r;
            if (fabsf(phase_diff) > 0.1f) {
                // Enhance existing phase differences slightly
                phase_l += 0.1f * width * phase_diff;
                phase_r -= 0.1f * width * phase_diff;
            }
            
            // Apply magnitude enhancements
            if (mag_l != mag_r) {
                // Enhance existing magnitude differences slightly
                float mag_ratio = mag_l / (mag_r + 1e-10f);
                if (mag_ratio > 1.0f) {
                    mag_l *= (1.0f + 0.2f * width * (mag_ratio - 1.0f));
                } else {
                    mag_r *= (1.0f + 0.2f * width * (1.0f/mag_ratio - 1.0f));
                }
            }
            
            // Convert back to complex representation
            real_l = mag_l * cosf(phase_l);
            imag_l = mag_l * sinf(phase_l);
            real_r = mag_r * cosf(phase_r);
            imag_r = mag_r * sinf(phase_r);
            
            // Store processed data
            proc->freq_data_l[bin] = real_l + I * imag_l;
            proc->freq_data_r[bin] = real_r + I * imag_r;
        }
    }
}

/**
 * Process a frame of audio (time and frequency domain)
 * with proper overlap buffer handling
 */
void process_frame(KStereoProcessor *proc, float *input_l, float *input_r, 
                   float *output_l, float *output_r, int num_samples) {
    int frame_start, i;
    int fft_size = proc->settings.fft_size;
    int hop_size = proc->hop_size;
    float *ambient_l, *ambient_r;
    float *temp_buffer_l, *temp_buffer_r;
    
    // Allocate temporary buffers for intermediate processing
    temp_buffer_l = (float*)calloc(num_samples, sizeof(float));
    if (!temp_buffer_l) {
        fprintf(stderr, "Memory allocation failed in process_frame\n");
        exit(1);
    }
    
    temp_buffer_r = (float*)calloc(num_samples, sizeof(float));
    if (!temp_buffer_r) {
        fprintf(stderr, "Memory allocation failed in process_frame\n");
        exit(1);
    }
    
    // Allocate ambient buffers
    ambient_l = (float*)calloc(num_samples, sizeof(float));
    if (!ambient_l) {
        fprintf(stderr, "Memory allocation failed in process_frame\n");
        exit(1);
    }
    
    ambient_r = (float*)calloc(num_samples, sizeof(float));
    if (!ambient_r) {
        fprintf(stderr, "Memory allocation failed in process_frame\n");
        exit(1);
    }
    
    // Clear output buffers
    memset(output_l, 0, sizeof(float) * num_samples);
    memset(output_r, 0, sizeof(float) * num_samples);
    
    // Choose ambience extraction method based on settings
    if (proc->settings.use_enhanced_ambience) {
        // Use the enhanced ambience extraction
        extract_enhanced_ambience(proc, input_l, input_r, ambient_l, ambient_r, num_samples);
    } else {
        // Use the original delay-based ambience extraction
        for (i = 0; i < num_samples; i++) {
            float amb_l, amb_r;
            process_delay_network(proc, input_l[i], input_r[i], &amb_l, &amb_r);
            ambient_l[i] = amb_l;
            ambient_r[i] = amb_r;
        }
    }
    
    // Mix ambient signal with original
    for (i = 0; i < num_samples; i++) {
        temp_buffer_l[i] = input_l[i] + ambient_l[i];
        temp_buffer_r[i] = input_r[i] + ambient_r[i];
    }
    
    // Process frames with overlap-add
    for (frame_start = -fft_size + hop_size; frame_start < num_samples; frame_start += hop_size) {
        // Fill input buffers with zeros
        memset(proc->input_buffer_l, 0, sizeof(float) * fft_size);
        memset(proc->input_buffer_r, 0, sizeof(float) * fft_size);
        
        // Copy valid samples to input buffers from time-domain processed samples
        for (i = 0; i < fft_size; i++) {
            int input_index = frame_start + i;
            if (input_index >= 0 && input_index < num_samples) {
                proc->input_buffer_l[i] = temp_buffer_l[input_index];
                proc->input_buffer_r[i] = temp_buffer_r[input_index];
            }
        }
        
        // Apply analysis window
        apply_window(proc->input_buffer_l, proc->analysis_window, fft_size);
        apply_window(proc->input_buffer_r, proc->analysis_window, fft_size);
        
        // Forward FFT
        fftwf_execute(proc->forward_plan_l);
        fftwf_execute(proc->forward_plan_r);
        
        // Analyze stereo correlation
        analyze_stereo_correlation(proc);
        
        // Enhance ambience in frequency domain
        enhance_ambience(proc);
        
        // Inverse FFT
        fftwf_execute(proc->inverse_plan_l);
        fftwf_execute(proc->inverse_plan_r);
        
        // Scale output and apply synthesis window
        for (i = 0; i < fft_size; i++) {
            proc->output_buffer_l[i] /= fft_size;
            proc->output_buffer_l[i] *= proc->synthesis_window[i];
            
            proc->output_buffer_r[i] /= fft_size;
            proc->output_buffer_r[i] *= proc->synthesis_window[i];
        }
        
        // FIXED: Proper overlap-add operation using the overlap buffer
        // Add to overlap buffers
        for (i = 0; i < fft_size; i++) {
            proc->overlap_buffer_l[i] += proc->output_buffer_l[i];
            proc->overlap_buffer_r[i] += proc->output_buffer_r[i];
        }
        
        // Copy completed samples to output and shift overlap buffer
        for (i = 0; i < hop_size && frame_start + i < num_samples; i++) {
            int output_index = frame_start + i;
            if (output_index >= 0) {
                output_l[output_index] = proc->overlap_buffer_l[i];
                output_r[output_index] = proc->overlap_buffer_r[i];
            }
        }
        
        // Shift overlap buffers
        memmove(proc->overlap_buffer_l, proc->overlap_buffer_l + hop_size, 
                sizeof(float) * (fft_size - hop_size));
        memmove(proc->overlap_buffer_r, proc->overlap_buffer_r + hop_size, 
                sizeof(float) * (fft_size - hop_size));
        memset(proc->overlap_buffer_l + (fft_size - hop_size), 0, sizeof(float) * hop_size);
        memset(proc->overlap_buffer_r + (fft_size - hop_size), 0, sizeof(float) * hop_size);
    }
    
    // Free temporary buffers
    free(temp_buffer_l);
    free(temp_buffer_r);
    free(ambient_l);
    free(ambient_r);
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
 * Process an entire audio file with proper pre/post roll handling
 */
void process_file(SNDFILE *infile, SNDFILE *outfile, SF_INFO *sfinfo, KStereoSettings *settings) {
    KStereoProcessor proc;
    int frames_read, i;
    int buffer_size = 16384; // Process in reasonably sized chunks
    
    // Initialize the processor
    initialize_processor(&proc, settings, sfinfo->samplerate);
    
    // Reset extraction state
    reset_ambience_extraction_state(&proc);
    
    // Calculate pre/post roll size for smooth transitions at file boundaries
    int pre_post_samples = (int)(0.1 * sfinfo->samplerate); // 100ms
    
    // Allocate larger buffers to handle pre/post roll
    float *input_buffer = (float*)malloc(sizeof(float) * (buffer_size + pre_post_samples * 2) * sfinfo->channels);
    float *output_buffer = (float*)malloc(sizeof(float) * (buffer_size + pre_post_samples * 2) * sfinfo->channels);
    float *input_left = (float*)malloc(sizeof(float) * (buffer_size + pre_post_samples * 2));
    float *input_right = (float*)malloc(sizeof(float) * (buffer_size + pre_post_samples * 2));
    float *output_left = (float*)malloc(sizeof(float) * (buffer_size + pre_post_samples * 2));
    float *output_right = (float*)malloc(sizeof(float) * (buffer_size + pre_post_samples * 2));
    
    if (!input_buffer || !output_buffer || !input_left || !input_right || !output_left || !output_right) {
        fprintf(stderr, "Memory allocation failed in process_file\n");
        exit(1);
    }
    
    // Clear pre-roll area
    memset(input_left, 0, sizeof(float) * pre_post_samples);
    memset(input_right, 0, sizeof(float) * pre_post_samples);
    
    // Read first block
    frames_read = sf_readf_float(infile, input_buffer, buffer_size);
    
    // Deinterleave with pre-roll padding
    for (i = 0; i < frames_read; i++) {
        input_left[pre_post_samples + i] = input_buffer[i * 2];
        input_right[pre_post_samples + i] = input_buffer[i * 2 + 1];
    }
    
    // Process first block with pre-roll silence
    process_frame(&proc, input_left, input_right, output_left, output_right, pre_post_samples + frames_read);
    
    // Only write actual data (skip pre-roll)
    for (i = 0; i < frames_read; i++) {
        output_buffer[i * 2] = output_left[pre_post_samples + i];
        output_buffer[i * 2 + 1] = output_right[pre_post_samples + i];
    }
    sf_writef_float(outfile, output_buffer, frames_read);
    
    // Process remaining full blocks - using normal processing
    while ((frames_read = sf_readf_float(infile, input_buffer, buffer_size)) > 0) {
        // De-interleave input
        for (i = 0; i < frames_read; i++) {
            input_left[i] = input_buffer[i * 2];
            input_right[i] = input_buffer[i * 2 + 1];
        }
        
        // Process this frame
        process_frame(&proc, input_left, input_right, output_left, output_right, frames_read);
        
        // Re-interleave output
        for (i = 0; i < frames_read; i++) {
            output_buffer[i * 2] = output_left[i];
            output_buffer[i * 2 + 1] = output_right[i];
        }
        
        // Write output
        sf_writef_float(outfile, output_buffer, frames_read);
    }
    
    // Process post-roll to ensure smooth tail
    // Fill post-roll buffers with silence
    memset(input_left, 0, sizeof(float) * (buffer_size + pre_post_samples * 2));
    memset(input_right, 0, sizeof(float) * (buffer_size + pre_post_samples * 2));
    memset(output_left, 0, sizeof(float) * (buffer_size + pre_post_samples * 2));
    memset(output_right, 0, sizeof(float) * (buffer_size + pre_post_samples * 2));
    
    // Process post-roll (include pre_post_samples at end to fully flush overlap buffers)
    process_frame(&proc, input_left, input_right, output_left, output_right, pre_post_samples * 2);
    
    // Write only the actual post-roll data (first half of the processed data)
    for (i = 0; i < pre_post_samples; i++) {
        output_buffer[i * 2] = output_left[i];
        output_buffer[i * 2 + 1] = output_right[i];
    }
    sf_writef_float(outfile, output_buffer, pre_post_samples);
    
    // Clean up
    cleanup_processor(&proc);
    free(input_buffer);
    free(output_buffer);
    free(input_left);
    free(input_right);
    free(output_left);
    free(output_right);
}

/**
 * Clean up processor resources
 */
void cleanup_processor(KStereoProcessor *proc) {
    // Free FFT resources
    fftwf_destroy_plan(proc->forward_plan_l);
    fftwf_destroy_plan(proc->inverse_plan_l);
    fftwf_destroy_plan(proc->forward_plan_r);
    fftwf_destroy_plan(proc->inverse_plan_r);
    
    fftwf_free(proc->input_buffer_l);
    fftwf_free(proc->output_buffer_l);
    fftwf_free(proc->input_buffer_r);
    fftwf_free(proc->output_buffer_r);
    fftwf_free(proc->freq_data_l);
    fftwf_free(proc->freq_data_r);
    
    // Free window resources
    free(proc->analysis_window);
    free(proc->synthesis_window);
    
    // Free overlap buffers
    free(proc->overlap_buffer_l);
    free(proc->overlap_buffer_r);
    
    // Free analysis resources
    free(proc->phase_correlation);
    free(proc->freq_width_factors);
    free(proc->prev_coherence);
    
    // Free ambience extraction resources
    fftwf_destroy_plan(proc->fft_forward_l);
    fftwf_destroy_plan(proc->fft_forward_r);
    fftwf_destroy_plan(proc->fft_inverse_l);
    fftwf_destroy_plan(proc->fft_inverse_r);
    fftwf_free(proc->temp_input_l);
    fftwf_free(proc->temp_input_r);
    fftwf_free(proc->temp_output_l);
    fftwf_free(proc->temp_output_r);
    fftwf_free(proc->temp_freq_l);
    fftwf_free(proc->temp_freq_r);
    
    // Free delay network
    free(proc->delay_network.delay_buffers_l);
    free(proc->delay_network.delay_buffers_r);
    free(proc->delay_network.delay_lengths);
    free(proc->delay_network.delay_gains);
    
    // Free frame history
    if (proc->frame_history.initialized) {
        fftwf_free(proc->frame_history.prev_frame_freq_l);
        fftwf_free(proc->frame_history.prev_frame_freq_r);
        free(proc->frame_history.prev_ambient_l);
        free(proc->frame_history.prev_ambient_r);
        proc->frame_history.initialized = false;
    }
}

/**
 * Parse command line arguments
 */
void parse_arguments(int argc, char **argv, KStereoSettings *settings) {
    int i;
    
    for (i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-crossfeed") == 0 && i+1 < argc) {
            settings->crossfeed = atof(argv[++i]);
            if (settings->crossfeed < 0.0f) settings->crossfeed = 0.0f;
            if (settings->crossfeed > 1.0f) settings->crossfeed = 1.0f;
        }
        else if (strcmp(argv[i], "-ambience") == 0 && i+1 < argc) {
            settings->ambience = atof(argv[++i]);
            if (settings->ambience < 0.0f) settings->ambience = 0.0f;
            if (settings->ambience > 1.0f) settings->ambience = 1.0f;
        }
        else if (strcmp(argv[i], "-depth") == 0 && i+1 < argc) {
            settings->depth = atof(argv[++i]);
            if (settings->depth < 0.0f) settings->depth = 0.0f;
            if (settings->depth > 1.0f) settings->depth = 1.0f;
        }
        else if (strcmp(argv[i], "-width") == 0 && i+1 < argc) {
            settings->width = atof(argv[++i]);
            if (settings->width < 0.0f) settings->width = 0.0f;
            if (settings->width > 1.0f) settings->width = 1.0f;
        }
        else if (strcmp(argv[i], "-lr") == 0) {
            settings->use_ms = 0;
        }
        else if (strcmp(argv[i], "-debug") == 0) {
            settings->debug_mode = 1;
        }
        else if (strcmp(argv[i], "-basic") == 0) {
            settings->use_enhanced_ambience = 0;
        }
        else if (strcmp(argv[i], "-enhanced") == 0) {
            settings->use_enhanced_ambience = 1;
        }
        else if (strcmp(argv[i], "-decorr") == 0 && i+1 < argc) {
            settings->ambience_decorr = atof(argv[++i]);
            if (settings->ambience_decorr < 0.0f) settings->ambience_decorr = 0.0f;
            if (settings->ambience_decorr > 1.0f) settings->ambience_decorr = 1.0f;
        }
        else if (strcmp(argv[i], "-coherence") == 0 && i+1 < argc) {
            settings->coherence_threshold = atof(argv[++i]);
            if (settings->coherence_threshold < 0.0f) settings->coherence_threshold = 0.0f;
            if (settings->coherence_threshold > 1.0f) settings->coherence_threshold = 1.0f;
        }
    }
}

/**
 * Print current settings
 */
void print_settings(KStereoSettings *settings) {
    printf("K-Stereo Processor Settings:\n");
    printf("----------------------------\n");
    printf("Processing Mode: %s\n", settings->use_ms ? "Mid/Side" : "Left/Right");
    printf("FFT Size: %d bins\n", settings->fft_size);
    printf("Crossfeed Amount: %.1f%%\n", settings->crossfeed * 100.0f);
    printf("Ambience Recovery: %.1f%%\n", settings->ambience * 100.0f);
    printf("Depth Enhancement: %.1f%%\n", settings->depth * 100.0f);
    printf("Width Enhancement: %.1f%%\n", settings->width * 100.0f);
    printf("Ambience Algorithm: %s\n", 
           settings->use_enhanced_ambience ? "Enhanced (PCA-based)" : "Basic (Delay-based)");
    if (settings->use_enhanced_ambience) {
        printf("Decorrelation Amount: %.1f%%\n", settings->ambience_decorr * 100.0f);
        printf("Coherence Threshold: %.1f%%\n", settings->coherence_threshold * 100.0f);
    }
    printf("----------------------------\n");
}
