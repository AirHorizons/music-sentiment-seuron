// Keys constants
var KEY_WIDTH  = 25;
var KEY_HEIGHT = 100;

// Keyboard constants
var KEYBOARD_OCTAVES      = 7;
var KEYBOARD_FIRST_OCTAVE = 1;

// Simulation constants
var MAX_VELOCITY = 128;

// ----------------------------------------
// SETUP FUNCTIONS
// ----------------------------------------

function setup() {
    // Create canvas
    canvas = createCanvas(windowWidth - 20, windowHeight - 20);

    setupKeyboard();
    setupSampler();
    setupMenu();

    isMousePressed = false;
    isMainStarted  = false;
    isAIPlaying    = false;

    genSequenceLen = 50;

    playingTimer    = 0;
    aiPlayingTimer  = 0;
    lastMeasureTime = 0;
    generationTime  = 2;

    baseTimeDiv = 16;

    keysPlayed = [];
    keysPlayedInTimeDiv = {};
}

function setupMenu() {
    keyboard.draw();
    background(20, 20, 20, 220);

    button = createButton('PLAY');
    button.position(width/2 - 100, height/2 - 50, 200, 100);
    button.mousePressed(function() {
        // Remove any HTML elements
        removeElements();

        // Change background color
        background(20, 20, 20, 255);

        // Start MAIN app
        isMainStarted = true;

        // Make ure tone.js is running
        if (Tone.context.state !== 'running') {
            Tone.context.resume();
        }

        // Start the tone.js scheduler
        Tone.Transport.bpm.value = 120;
        Tone.Transport.start();
    });
    button.id("play-button");
}

function setupKeyboard() {
    // Create keyboard
    var kX = width/2 - (KEYBOARD_OCTAVES * KEY_WIDTH * N_WHITE_KEY)/2;
    var kY = height - KEY_HEIGHT;

    keyboard = new Keyboard(kX, kY, KEY_WIDTH, KEY_HEIGHT, KEYBOARD_OCTAVES, KEYBOARD_FIRST_OCTAVE);

    keyboard.keyDown = function(pitch, velocity) {
        playingTimer = 0;

        note = Tone.Frequency(pitch, "midi").toNote();
        sampler.triggerAttack(note, Tone.Transport.now(), velocity);

        if(!isAIPlaying) {
            keysPlayedInTimeDiv[pitch] = {};
            keysPlayedInTimeDiv[pitch].pitch = pitch;
            keysPlayedInTimeDiv[pitch].start = Tone.Transport.seconds;
        }
    }

    keyboard.keyUp = function(pitch, duration) {
        playingTimer = 0;

        note = Tone.Frequency(pitch, "midi").toNote();
        sampler.triggerRelease(note);

        if(!isAIPlaying) {
            if (pitch in keysPlayedInTimeDiv) {
                keysPlayedInTimeDiv[pitch].end = Tone.Transport.seconds;
                keysPlayed.push(keysPlayedInTimeDiv[pitch]);
            }
        }
    }
}

function setupSampler() {
    // Create sampler with salamder samples
    sampler = new Tone.Sampler({
        "A0" : "A0.[mp3|ogg]",
        "C1" : "C1.[mp3|ogg]",
        "D#1" : "Ds1.[mp3|ogg]",
        "F#1" : "Fs1.[mp3|ogg]",
        "A1" : "A1.[mp3|ogg]",
        "C2" : "C2.[mp3|ogg]",
        "D#2" : "Ds2.[mp3|ogg]",
        "F#2" : "Fs2.[mp3|ogg]",
        "A2" : "A2.[mp3|ogg]",
        "C3" : "C3.[mp3|ogg]",
        "D#3" : "Ds3.[mp3|ogg]",
        "F#3" : "Fs3.[mp3|ogg]",
        "A3" : "A3.[mp3|ogg]",
        "C4" : "C4.[mp3|ogg]",
        "D#4" : "Ds4.[mp3|ogg]",
        "F#4" : "Fs4.[mp3|ogg]",
        "A4" : "A4.[mp3|ogg]",
        "C5" : "C5.[mp3|ogg]",
        "D#5" : "Ds5.[mp3|ogg]",
        "F#5" : "Fs5.[mp3|ogg]",
        "A5" : "A5.[mp3|ogg]",
        "C6" : "C6.[mp3|ogg]",
        "D#6" : "Ds6.[mp3|ogg]",
        "F#6" : "Fs6.[mp3|ogg]",
        "A6" : "A6.[mp3|ogg]",
        "C7" : "C7.[mp3|ogg]",
        "D#7" : "Ds7.[mp3|ogg]",
        "F#7" : "Fs7.[mp3|ogg]",
        "A7" : "A7.[mp3|ogg]",
        "C8" : "C8.[mp3|ogg]"
    }, {
        "release" : 1,
        "baseUrl" : "./static/audio/salamander/"
    }).toMaster();
}

function startAIPart(part) {
    var evIx = 0;

    Tone.Transport.scheduleRepeat(function(time) {
        if(!isAIPlaying)
            return;

        if(evIx < part.length) {
            if(part[evIx][0] == ".") {
                evIx++;
            }
            else if(part[evIx][0] == "n" || part[evIx][0] == "t") {
                while(evIx < part.length && part[evIx][0] != ".") {
                    var ev = part[evIx].split("_");

                    if(part[evIx][0] == "n") {
                        var pitch    = parseInt(ev[1]);
                        var duration = ev[2] + "n";
                        var velocity = parseInt(ev[3])/MAX_VELOCITY;

                        console.log("PLAY NOTE: " + part[evIx]);
                        keyboard.playKeyWithPitch(pitch, Tone.Time(duration).toSeconds(), velocity);
                    }
                    else {
                        var tempo = parseInt(ev[1]);
                        if(tempo >= 20) {
                            Tone.Transport.bpm.value = tempo;
                        }
                    }

                    evIx++;
                }
            }
        }
        else {
            console.log("AI STOP!");
            isAIPlaying = false;

            Tone.Transport.cancel();
            background(20, 20, 20, 255);
        }
    }, baseTimeDiv.toString() + "n");
}

// ----------------------------------------
// DRAWING FUNCTIONS
// ----------------------------------------

function draw() {
    if(isMainStarted) {
        var dt = (window.performance.now() - canvas._pInst._lastFrameTime)/1000;

        update(dt);

        keyboard.update(dt, isMousePressed);
        keyboard.draw();
    }
}

// ----------------------------------------
// UPDATE FUNCTIONS
// ----------------------------------------

function update(dt) {
    playingTimer += dt;
    if(playingTimer > generationTime) {
        if(keysPlayed.length > 0 && !isAIPlaying) {
            console.log("AI PLAY!");
            noteSequence = keys2NoteSequence(keysPlayed);
            console.log(noteSequence);

            isAIPlaying = true;
            aiPlayingTimer = 0;

            sendNoteSequence(noteSequence, genSequenceLen, generationCallback);

            keysPlayed = [];
            keysPlayedInTimeDiv = {};
        }

        playingTimer = 0;
    }

    if(isAIPlaying) {
        // Animate background
        var from = color(20, 20, 20, 255);
        var to   = color(80, 80, 80, 255);

        aiPlayingTimer += dt * 5;
        var final = lerpColor(from, to, sin(aiPlayingTimer));

        background(final);
    }
}

// ----------------------------------------
// CALLBACK FUNCTIONS
// ----------------------------------------

function generationCallback(generatedSequence) {
    var part = generatedSequence.split(" ");
    startAIPart(part);
}

function keys2NoteSequence(keysPlayed) {
    var lastNoteStart = 1000;

    var noteSequence = ""
    for(var i = 0; i < keysPlayed.length; i++) {
        var duration = Math.round((keysPlayed[i].end - keysPlayed[i].start) * 100)/100;
        duration = Tone.Time(duration).toNotation()[0];
        noteSequence += "n_" + keysPlayed[i].pitch + "_" + duration + "_80";

        notesDist = Math.abs(keysPlayed[i].start - lastNoteStart);
        if(notesDist > Tone.Time("16n").toSeconds()) {
            noteSequence += " . ";
        }
        else {
            noteSequence += " ";
        }

        lastNoteStart = keysPlayed[i].start;
    }

    return noteSequence.substring(0, noteSequence.length - 1);
}

// ----------------------------------------
// IO FUNCTIONS
// ----------------------------------------

function mousePressed() {
    if(!isAIPlaying) {
        isMousePressed = true;
    }
}

function mouseReleased() {
    isMousePressed = false;
}

function keyPressed() {
    if(!isAIPlaying) {
        keyboard.keyPressed(keyCode);
    }
}

function keyReleased() {
    keyboard.keyReleased(keyCode);
}

function windowResized() {
  resizeCanvas(windowWidth - 20, windowHeight - 20);
  button.position(width/2 - 100, height/2 - 50, 200, 100);
}
