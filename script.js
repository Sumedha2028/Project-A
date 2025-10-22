script.js 

    // --- 1. DOM Element References ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const recordBtn = document.getElementById('record-btn');
    const timerDisplay = document.getElementById('timer');
    const statusMessage = document.getElementById('status-message');
    
    const controlSection = document.getElementById('control-section');
    const waveformEl = document.getElementById('waveform');
    const spectrogramEl = document.getElementById('spectrogram');
    const classifyBtn = document.getElementById('classify-btn');
    
    const resultsSection = document.getElementById('results-section');
    const loader = document.getElementById('loader');
    const resultsContent = document.getElementById('results-content');
    const predictedGenreEl = document.getElementById('predicted-genre');
    const chartContainer = document.getElementById('chart-container');

    // --- 2. Constants & State Variables ---

    // !!!!!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!!!!!!!
    // You MUST replace this with the path to your hosted model.json file.
    // This model is ASSUMED to be a GTZAN-trained model.
    const MODEL_URL = './model/model.json'; // <--- REPLACE THIS
    
    // You MUST replace these with the exact specs your model expects.
    // These are common values for Mel Spectrogram models (e.g., [1, 96, 64, 1])
    const MODEL_EXPECTED_SAMPLE_RATE = 22050;
    const MODEL_INPUT_NUM_MELS = 64;   // Number of Mel bins
    const MODEL_INPUT_NUM_FRAMES = 96; // Number of time-frames
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    const GENRE_CLASSES = [
        'blues', 'classical', 'country', 'disco', 'hiphop', 
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ];
    
    // Application State
    let model;
    let audioContext;
    let wavesurfer;
    let currentAudioBuffer;
    let essentiaWorker;
    
    // Recording State
    let mediaRecorder;
    let audioChunks = [];
    let timerInterval;
    let secondsElapsed = 0;


    // --- 3. Web Worker for Feature Extraction ---
    // This code runs in a separate thread to avoid freezing the UI.
    // It's defined as a string and converted to a Blob URL.
    
    const workerScript = `
        // We need to import the Essentia.js core script in the worker
        importScripts('https://cdn.jsdelivr.net/npm/essentia.js@0.1.3/dist/essentia.js-core.js');

        let essentia;

        // Worker message handler
        onmessage = async (event) => {
            const { type, audioData, sampleRate, targetSr, numMels, numFrames } = event.data;

            if (type === 'init') {
                // Load the WASM module
                const EssentiaWASM = await import(event.data.wasmPath);
                essentia = new Essentia(EssentiaWASM.EssentiaWASM);
                // Send ready message back to main thread
                postMessage({ type: 'ready' });
            
            } else if (type === 'process') {
                try {
                    // 1. Convert audio data (which is a Float32Array) to Essentia's vector format
                    const audioVector = essentia.arrayToVector(audioData);
                    
                    // 2. Resample if necessary
                    let resampledVector = audioVector;
                    if (sampleRate !== targetSr) {
                        resampledVector = essentia.Resample(audioVector, sampleRate, targetSr).signal;
                    }

                    // 3. Compute Mel Spectrogram
                    // Config: 2048 frame size, 1024 hop size, 64 mel bands
                    const melSpec = essentia.MelSpectrogram(resampledVector, 
                        targetSr,     // sampleRate
                        2048,         // frameSize
                        1024,         // hopSize
                        numMels,      // numberBands
                        'triangle',   // type
                        'unit_area',  // areaType
                        0,            // lowBound
                        targetSr / 2, // highBound
                        0.5           // htk
                    );

                    // 4. Convert to a standard 2D array and apply log compression (common for mel specs)
                    const melSpecGrid = [];
                    for (let i = 0; i < melSpec.melbands.size(); i++) {
                        const frame = essentia.vectorToArray(melSpec.melbands.get(i));
                        // Apply log: 10 * log10(value + epsilon)
                        const logFrame = frame.map(val => 10 * Math.log10(val + 1e-6));
                        melSpecGrid.push(logFrame);
                    }
                    
                    // 5. Pad or Truncate to the model's expected number of frames (MODEL_INPUT_NUM_FRAMES)
                    const processedGrid = padOrTruncate(melSpecGrid, numFrames, numMels);
                    
                    // 6. Send the processed features back to the main thread
                    postMessage({ type: 'result', features: processedGrid });

                } catch (e) {
                    postMessage({ type: 'error', message: e.message });
                }
            }
        };

        // Utility function inside the worker
        function padOrTruncate(grid, numFrames, numMels) {
            const currentFrames = grid.length;
            
            if (currentFrames > numFrames) {
                // Truncate (simple slice)
                return grid.slice(0, numFrames);
            } else if (currentFrames < numFrames) {
                // Pad with zeros (or minimum value)
                const paddedGrid = [...grid];
                const padding = Array(numMels).fill(-60); // Padding with log(1e-6)
                
                while (paddedGrid.length < numFrames) {
                    paddedGrid.push(padding);
                }
                return paddedGrid;
            }
            return grid;
        }
    `;

    // --- 4. Initialization Functions ---

    /**
     * Main function to initialize the application
     */
    function init() {
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) {
            statusMessage.textContent = 'Error: Web Audio API is not supported in this browser.';
            return;
        }

        initWavesurfer();
        initWorker();
        loadModel();
        setupEventListeners();
    }

    /**
     * Initializes the Essentia.js Web Worker
     */
    function initWorker() {
        const workerBlob = new Blob([workerScript], { type: 'application/javascript' });
        essentiaWorker = new Worker(URL.createObjectURL(workerBlob));
        
        // Pass the WASM module path to the worker for it to initialize
        essentiaWorker.postMessage({
            type: 'init',
            wasmPath: 'https://cdn.jsdelivr.net/npm/essentia.js@0.1.3/dist/essentia-wasm.module.js'
        });

        // Handle messages from the worker
        essentiaWorker.onmessage = (event) => {
            const { type } = event.data;
            if (type === 'ready') {
                console.log('Essentia.js Worker is ready.');
                updateStatus('Model loaded. Essentia.js ready. Select a file or record.');
            } else if (type === 'result') {
                // Feature extraction is done, run model inference
                runInference(event.data.features);
            } else if (type === 'error') {
                console.error('Error from Essentia.js worker:', event.data.message);
                setLoading(false, 'Feature extraction failed.');
            }
        };
    }

    /**
     * Loads the TensorFlow.js model
     */
    async function loadModel() {
        try {
            model = await tf.loadGraphModel(MODEL_URL);
            await model.predict(tf.zeros([1, MODEL_INPUT_NUM_FRAMES, MODEL_INPUT_NUM_MELS, 1])).dispose(); // Warm-up
            
            console.log('Model loaded successfully.');
            if (essentiaWorker) { // Check if worker is also ready
                updateStatus('Model loaded. Essentia.js ready. Select a file or record.');
            } else {
                updateStatus('Model loaded. Waiting for audio processor...');
            }
        } catch (e) {
            console.error(e);
            updateStatus('Error: Could not load the ML model.');
        }
    }

    /**
     * Initializes Wavesurfer.js with Spectrogram plugin
     */
    function initWavesurfer() {
        wavesurfer = WaveSurfer.create({
            container: waveformEl,
            waveColor: '#ddd',
            progressColor: var(--primary-color, '#4a90e2'),
            barWidth: 3,
            barRadius: 3,
            cursorWidth: 1,
            cursorColor: '#fff',
            plugins: [
                Spectrogram.create({
                    container: spectrogramEl,
                    labels: true,
                    fftSamples: 1024,
                    colorMap: 'viridis'
                }),
            ],
        });

        wavesurfer.on('ready', () => {
            controlSection.classList.remove('hidden');
            classifyBtn.disabled = false;
        });

        wavesurfer.on('error', (e) => {
            console.error(e);
            updateStatus('Error: Could not load audio file.');
        });
    }

    /**
     * Sets up all user event listeners
     */
    function setupEventListeners() {
        // Drag and Drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
        });
        dropZone.addEventListener('drop', handleDrop, false);

        // File Input
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleFile(e.target.files[0]);
            }
        });

        // Record Button
        recordBtn.addEventListener('click', toggleRecording);

        // Classify Button
        classifyBtn.addEventListener('click', handleClassify);
    }


    // --- 5. Event Handlers ---

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        if (file && file.type.startsWith('audio/')) {
            handleFile(file);
        } else {
            updateStatus('Error: Please drop a valid audio file.');
        }
    }

    /**
     * Handles file input (from drag/drop or browse)
     * @param {File} file
     */
    async function handleFile(file) {
        if (!file) return;

        // Reset UI
        resetResults();
        updateStatus('Loading audio file...');

        try {
            const arrayBuffer = await file.arrayBuffer();
            // Decode audio data for feature extraction
            currentAudioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Load into Wavesurfer for visualization
            wavesurfer.load(URL.createObjectURL(file));
            
            updateStatus('Audio loaded. Ready to classify.');
        } catch (e) {
            console.error(e);
            updateStatus('Error: Could not decode audio file.');
        }
    }

    /**
     * Toggles audio recording on/off
     */
    async function toggleRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            // Stop recording
            mediaRecorder.stop();
            recordBtn.classList.remove('recording');
            recordBtn.innerHTML = '<span class="icon">🎤</span> Record Audio';
            
            // Stop timer
            clearInterval(timerInterval);
            secondsElapsed = 0;

        } else {
            // Start recording
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (e) => {
                    audioChunks.push(e.data);
                };

                mediaRecorder.onstop = () => {
                    // Combine audio chunks into a Blob
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    // Pass the Blob (as a File object) to handleFile
                    handleFile(new File([audioBlob], 'recording.wav', { type: 'audio/wav' }));
                    
                    // Clean up the stream tracks
                    stream.getTracks().forEach(track => track.stop());
                };

                mediaRecorder.start();
                recordBtn.classList.add('recording');
                recordBtn.innerHTML = '<span class="icon">🛑</span> Stop Recording';
                
                // Start timer
                timerDisplay.textContent = '00:00';
                secondsElapsed = 0;
                timerInterval = setInterval(() => {
                    secondsElapsed++;
                    const minutes = Math.floor(secondsElapsed / 60).toString().padStart(2, '0');
                    const seconds = (secondsElapsed % 60).toString().padStart(2, '0');
                    timerDisplay.textContent = `${minutes}:${seconds}`;
                }, 1000);

            } catch (e) {
                console.error(e);
                updateStatus('Error: Could not access microphone.');
            }
        }
    }

    /**
     * Handles the "Classify" button click
     */
    function handleClassify() {
        if (!currentAudioBuffer || !model) {
            updateStatus('Error: Audio or model not ready.');
            return;
        }

        setLoading(true, 'Extracting features...');
        
        // Send audio data to the Web Worker for processing
        essentiaWorker.postMessage({
            type: 'process',
            audioData: currentAudioBuffer.getChannelData(0), // Send single channel (mono)
            sampleRate: currentAudioBuffer.sampleRate,
            targetSr: MODEL_EXPECTED_SAMPLE_RATE,
            numMels: MODEL_INPUT_NUM_MELS,
            numFrames: MODEL_INPUT_NUM_FRAMES
        });
    }


    // --- 6. ML Inference & Display ---

    /**
     * Runs inference after features are extracted by the worker
     * @param {number[][]} features - The [numFrames, numMels] feature grid
     */
    async function runInference(features) {
        if (!model) {
            console.error('Model not loaded.');
            setLoading(false, 'Error: Model not loaded.');
            return;
        }

        setLoading(true, 'Classifying...');

        let prediction;
        try {
            // Reshape the 2D array [numFrames, numMels] into the 4D tensor the model expects
            const inputTensor = tf.tensor(features)
                                  .reshape([1, MODEL_INPUT_NUM_FRAMES, MODEL_INPUT_NUM_MELS, 1]);

            // Run prediction
            prediction = model.predict(inputTensor);

            // Get the probabilities as a standard array
            const probabilities = await prediction.data();
            
            // Clean up tensors
            inputTensor.dispose();
            prediction.dispose();

            // Display the results
            displayResults(probabilities);

        } catch (e) {
            console.error('Error during inference:', e);
            setLoading(false, 'Inference failed.');
            if (prediction) prediction.dispose();
        }
    }

    /**
     * Displays the classification results in the UI
     * @param {Float32Array} probabilities - Array of probabilities from the model
     */
    function displayResults(probabilities) {
        // Map probabilities to their genre names
        const results = Array.from(probabilities)
            .map((prob, i) => ({
                genre: GENRE_CLASSES[i] || 'Unknown',
                probability: prob
            }))
            .sort((a, b) => b.probability - a.probability); // Sort descending

        // Get top genre
        const topGenre = results[0];
        predictedGenreEl.textContent = topGenre.genre;

        // Get top 5 for the chart
        const top5 = results.slice(0, 5);
        const labels = top5.map(r => r.genre);
        const values = top5.map(r => r.probability * 100); // As percentage

        drawChart(labels, values);
        setLoading(false);
    }

    /**
     * Draws the Plotly.js bar chart
     * @param {string[]} labels - Genre names
     * @param {number[]} values - Confidence percentages
     */
    function drawChart(labels, values) {
        const data = [{
            type: 'bar',
            x: values,
            y: labels,
            orientation: 'h',
            text: values.map(v => v.toFixed(1) + '%'),
            textposition: 'auto',
            marker: {
                color: var(--primary-color, '#4a90e2')
            }
        }];

        const layout = {
            title: 'Top 5 Predictions',
            xaxis: {
                title: 'Confidence (%)',
                color: '#fff',
                range: [0, 100]
            },
            yaxis: {
                automargin: true,
                color: '#fff'
            },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: '#fff'
            },
            margin: { l: 100, r: 20, t: 40, b: 40 }
        };

        Plotly.newPlot(chartContainer, data, layout, { responsive: true });
    }

    // --- 7. UI Utility Functions ---

    /**
     * Updates the global status message
     * @param {string} message
     */
    function updateStatus(message) {
        statusMessage.textContent = message;
    }

    /**
     * Manages the loading state of the UI
     * @param {boolean} isLoading
     * @param {string} [message] - Optional message for the status
     */
    function setLoading(isLoading, message = '') {
        if (isLoading) {
            resultsSection.classList.remove('hidden');
            loader.classList.remove('hidden');
            resultsContent.classList.add('hidden');
            classifyBtn.disabled = true;
            updateStatus(message || 'Processing...');
        } else {
            loader.classList.add('hidden');
            resultsContent.classList.remove('hidden');
            classifyBtn.disabled = false;
            updateStatus(message || 'Classification complete.');
        }
    }

    /**
     * Resets the results section to its initial state
     */
    function resetResults() {
        resultsSection.classList.add('hidden');
        resultsContent.classList.add('hidden');
        loader.classList.add('hidden');
        predictedGenreEl.textContent = '...';
        Plotly.purge(chartContainer);
    }

    // --- 8. Start the application ---
    init();

});
