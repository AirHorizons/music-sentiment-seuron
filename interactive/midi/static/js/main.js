// Keyboard constants
var KEY_WIDTH  = 25;
var KEY_HEIGHT = 100;
var KEYBOARD_OCTAVES = 7;

// Simulation constants
var MAX_VELOCITY = 128;

// ----------------------------------------
// SETUP FUNCTIONS
// ----------------------------------------

function setup() {
    // Create canvas
    canvas = createCanvas(windowWidth - 20, windowHeight - 20);

    setupKeyboard();
    setupMenu();

    isMousePressed = false;
    isMainStarted  = false;
    isAIPlaying    = false;

    genSequenceLen = 400;

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

    keyboard = new Keyboard(kX, kY, KEY_WIDTH, KEY_HEIGHT, KEYBOARD_OCTAVES);

    keyboard.keyDown = function(pitch, velocity) {
        playingTimer = 0;

        if(!isAIPlaying) {
            keysPlayedInTimeDiv[pitch] = {};
            keysPlayedInTimeDiv[pitch].pitch = pitch;
            keysPlayedInTimeDiv[pitch].start = Tone.Transport.seconds;
        }
    }

    keyboard.keyUp = function(pitch, duration) {
        playingTimer = 0;

        if(!isAIPlaying) {
            if (pitch in keysPlayedInTimeDiv) {
                keysPlayedInTimeDiv[pitch].end = Tone.Transport.seconds;
                keysPlayed.push(keysPlayedInTimeDiv[pitch]);
            }
        }
    }
}

function startAIPart(part) {
    var evIx = 0;

    console.log(part)
    Tone.Transport.scheduleRepeat(function(time) {
        if(!isAIPlaying)
            return;

        var duration = "4n";
        var velocity = 1.0

        if(evIx < part.length) {
            if(part[evIx][0] == ".") {
                evIx++;
            }
            else {
                while(evIx < part.length && part[evIx][0] != ".") {
                    var ev = part[evIx].split("_");

                    switch (part[evIx][0]) {
                      case "n":
                        console.log("PLAY NOTE: " + part[evIx]);
                        var pitch = parseInt(ev[1]);
                        keyboard.playKeyWithPitch(pitch, Tone.Time(duration).toSeconds(), velocity);
                        break;
                      case "d":
                        duration = ev[1] + "n";
                        break;
                      case "v":
                        velocity = parseInt(ev[1])/MAX_VELOCITY;
                        break;
                      case "t":
                        var tempo = parseInt(ev[1]);
                        if(tempo >= 20) {
                            Tone.Transport.bpm.value = tempo;
                        }
                        break;
                    }

                    evIx++;
                }
            }
        }
        else {
            console.log("AI STOP!");
            isAIPlaying = false;

            Tone.Transport.bpm.value = 120;
            Tone.Transport.cancel();
            Tone.Transport.start();

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
    var lastNoteStart = 0;

    keysPlayed.sort(function (a, b) {
        return a.start - b.start;
    });

    var lastDuration = ""
    var noteSequence = "v_100 t_120 "

    for(var i = 0; i < keysPlayed.length; i++) {
        var duration = Math.round((keysPlayed[i].end - keysPlayed[i].start) * 100)/100;
        duration = Tone.Time(duration).toNotation()[0];

        if(duration != lastDuration) {
          noteSequence += "d_" + duration + " "
        }

        if(i > 0) {
            notesDist = Math.abs(keysPlayed[i].start - lastNoteStart);

            if(notesDist > Tone.Time("16n").toSeconds()) {
                nRests = Math.ceil(notesDist/Tone.Time("16n").toSeconds())
                for(var j = 0; j < nRests; j++) {
                    noteSequence += ". ";
                }
            }
        }

        noteSequence += "n_" + keysPlayed[i].pitch + " ";

        lastNoteStart = keysPlayed[i].start;
        lastDuration = duration
    }

    noteSequence += ".";
    return noteSequence;
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
