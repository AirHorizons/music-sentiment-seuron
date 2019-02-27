// Keys constants
var KEY_WIDTH  = 30;
var KEY_HEIGHT = 100;

// Keyboard constants
var KEYBOARD_X            = 0;
var KEYBOARD_Y            = KEY_HEIGHT + 22;
var KEYBOARD_OCTAVES      = 6;
var KEYBOARD_FIRST_OCTAVE = 2;

// Simulation constants
var SEQUENCE_SEED_TIME = 5;

function setup() {
    // Create canvas
    createCanvas(windowWidth - 20, windowHeight - 20);

    // Create keyboard
    keyboard = new Keyboard(KEYBOARD_X, windowHeight - KEYBOARD_Y, KEY_WIDTH, KEY_HEIGHT, KEYBOARD_OCTAVES, KEYBOARD_FIRST_OCTAVE)

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

    keysPressed = [];

    keyboard.keyDown = function(pitch) {
        keysPressed.push(pitch);
        note = Tone.Frequency(pitch, "midi").toNote();
        sampler.triggerAttack(note);
    }

    keyboard.keyUp = function(pitch) {
        note = Tone.Frequency(pitch, "midi").toNote();
        sampler.triggerRelease(note);
    }

    last = millis()
}

function update() {
    now = millis();

    if((now - last)/1000 > SEQUENCE_SEED_TIME) {
        sendNoteSequence(keysPressed.toString());

        keysPressed = [];
        last = millis();
    }
}

function draw() {
    update();

    keyboard.update();
    keyboard.draw();
}

function mousePressed() {
    if(keyboard.isMouseOver()) {
        keyboard.didMouseClick = true;
    }
}

function mouseReleased() {
    keyboard.didMouseClick = false;

    keyboard.releaseAllKeys();
    sampler.releaseAll();
}

function keyPressed() {
    switch (keyCode) {
        case 65: // A
            keyboard.octaves[2].whiteKeys[0].isKeyPressed = true;
            break;
        case 87: // W
            keyboard.octaves[2].blackKeys[0].isKeyPressed = true;
            break;
        case 83: // S
            keyboard.octaves[2].whiteKeys[1].isKeyPressed = true;
            break;
        case 69: // E
            keyboard.octaves[2].blackKeys[1].isKeyPressed = true;
            break;
        case 68: // D
            keyboard.octaves[2].whiteKeys[2].isKeyPressed = true;
            break;
        case 70: // F
            keyboard.octaves[2].whiteKeys[3].isKeyPressed = true;
            break;
        case 84: // T
            keyboard.octaves[2].blackKeys[2].isKeyPressed = true;
            break;
        case 71: // G
            keyboard.octaves[2].whiteKeys[4].isKeyPressed = true;
            break;
        case 89: // Y
            keyboard.octaves[2].blackKeys[3].isKeyPressed = true;
            break;
        case 72: // H
            keyboard.octaves[2].whiteKeys[5].isKeyPressed = true;
            break;
        case 85: // U
            keyboard.octaves[2].blackKeys[4].isKeyPressed = true;
            break;
        case 74: // J
            keyboard.octaves[2].whiteKeys[6].isKeyPressed = true;
            break;
        case 75: // k
            keyboard.octaves[3].whiteKeys[0].isKeyPressed = true;
            break;
        case 79: // O
            keyboard.octaves[3].blackKeys[0].isKeyPressed = true;
            break;
        case 76: // L
            keyboard.octaves[3].whiteKeys[1].isKeyPressed = true;
            break;
        case 80: // P
            keyboard.octaves[3].blackKeys[1].isKeyPressed = true;
            break;
        case 186: // ;
            keyboard.octaves[3].whiteKeys[2].isKeyPressed = true;
            break;
        case 222: // ;
            keyboard.octaves[3].whiteKeys[3].isKeyPressed = true;
            break;
        case 221: // ;
            keyboard.octaves[3].blackKeys[2].isKeyPressed = true;
            break;
        default:
            break;
    }
}

function keyReleased() {
    switch (keyCode) {
        case 65: // A
            keyboard.octaves[2].whiteKeys[0].isKeyPressed = false;
            break;
        case 87: // W
            keyboard.octaves[2].blackKeys[0].isKeyPressed = false;
            break;
        case 83: // S
            keyboard.octaves[2].whiteKeys[1].isKeyPressed = false;
            break;
        case 69: // E
            keyboard.octaves[2].blackKeys[1].isKeyPressed = false;
            break;
        case 68: // D
            keyboard.octaves[2].whiteKeys[2].isKeyPressed = false;
            break;
        case 70: // F
            keyboard.octaves[2].whiteKeys[3].isKeyPressed = false;
            break;
        case 84: // T
            keyboard.octaves[2].blackKeys[2].isKeyPressed = false;
            break;
        case 71: // G
            keyboard.octaves[2].whiteKeys[4].isKeyPressed = false;
            break;
        case 89: // Y
            keyboard.octaves[2].blackKeys[3].isKeyPressed = false;
            break;
        case 72: // H
            keyboard.octaves[2].whiteKeys[5].isKeyPressed = false;
            break;
        case 85: // U
            keyboard.octaves[2].blackKeys[4].isKeyPressed = false;
            break;
        case 74: // J
            keyboard.octaves[2].whiteKeys[6].isKeyPressed = false;
            break;
        case 75: // k
            keyboard.octaves[3].whiteKeys[0].isKeyPressed = false;
            break;
        case 79: // O
            keyboard.octaves[3].blackKeys[0].isKeyPressed = false;
            break;
        case 76: // L
            keyboard.octaves[3].whiteKeys[1].isKeyPressed = false;
            break;
        case 80: // P
            keyboard.octaves[3].blackKeys[1].isKeyPressed = false;
            break;
        case 186: // ;
            keyboard.octaves[3].whiteKeys[2].isKeyPressed = false;
            break;
        case 222: // ;
            keyboard.octaves[3].whiteKeys[3].isKeyPressed = false;
            break;
        case 221: // ;
            keyboard.octaves[3].blackKeys[2].isKeyPressed = false;
            break;
        default:
            break;
    }
}
