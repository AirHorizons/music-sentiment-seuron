var N_WHITE_KEY = 7;
var N_BLACK_KEY = 6;
var KEYS_PER_OCTAVE = (N_WHITE_KEY + N_BLACK_KEY - 2);

var MIDI_MAX_PITCH = 128;

var KeyColor= {
  WHITE: 0,
  BLACK: 1,
};

var KeyState = {
  RELEASED: 0,
  SELECTED: 1,
  PLAYING : 2,
};

class Key {
    constructor(x, y, w, h, pitch, color) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;

        this.pitch = pitch;
        this.velocity = 1;

        this.pressed = false;

        this.pressedTimer = 0;
        this.releaseTimer = 0;

        this.color = color;
        this.state = KeyState.RELEASED;
    }

    isKeyPressed() {
        return this.pressed;
    }

    isMouseOver() {
        var minX = this.x
        var maxX = this.x + this.w
        var minY = this.y
        var maxY = this.y + this.h

        if (mouseX >= minX && mouseX <= maxX && mouseY >= minY && mouseY <= maxY) {
            return true;
         }

         return false;
    }

    play() {
        if(this.state != KeyState.PLAYING) {
            this.keyboard.keyDown(this.pitch, this.velocity);
            this.state = KeyState.PLAYING;
        }
    }

    release() {
        if(this.state == KeyState.PLAYING)
            this.keyboard.keyUp(this.pitch, this.pressedTimer);
        this.state = KeyState.RELEASED;

        this.pressedTimer = 0;
        this.velocity = 1;
    }

    select() {
        if(this.state == KeyState.PLAYING)
            this.keyboard.keyUp(this.pitch, this.pressedTimer);

        this.state = KeyState.SELECTED;

        this.pressedTimer = 0;
    }

    update(dt, isMousePressed) {
        if(this.isMouseOver()) {
            isMousePressed ? this.play() : this.select();
        }

        if(this.isKeyPressed()) {
            this.play();

            if(this.releaseTimer > 0) {
                this.releaseTimer -= dt;

                if(this.releaseTimer <= 0) {
                    this.release();

                    this.pressed = false;
                    this.releaseTimer = 0;
                }
            }
        }

        if(!this.isMouseOver() && !this.isKeyPressed()) {
            this.release();
        }

        if(this.state == KeyState.PLAYING) {
            this.pressedTimer += dt;
        }
    }

    draw(releaseFillColor) {
        strokeWeight(2);

        // Set fill and stroke colors
        switch (this.state) {
            case KeyState.RELEASED:
                this.color == KeyColor.WHITE ? fill(50,50,50) : fill(10,10,10);
                stroke(220);
                break;
            case KeyState.SELECTED:
                fill(150,150,150);
                stroke(220);
                break;
            case KeyState.PLAYING:
                this.color == KeyColor.WHITE ?  fill(255,163,0,30) : fill(255,163,0);
                this.color == KeyColor.WHITE ? stroke(20,20,20,30) : stroke(220);
                break;
        }

        // Draw key as a rect
        rect(this.x, this.y, this.w, this.h);
    }
}

class Octave {
    constructor(keyboard, octaveIx, keyWidth, keyHeight) {
        this.ix = octaveIx;
        this.w = N_WHITE_KEY * keyWidth;
        this.h = keyHeight;
        this.x = keyboard.x + this.ix * this.w;
        this.y = keyboard.y;

        this.whiteKeys = []
        this.blackKeys = []

        this.keyboard = keyboard;
        this.keySelected = null;

        this.constructKeys(keyWidth, keyHeight);
    }

    constructKeys(keyWidth, keyHeight) {
        // Create white keys
        var whitePitch = ((this.keyboard.firstOctave + this.ix) * KEYS_PER_OCTAVE) + this.ix;
        for (var i = 0; i < N_WHITE_KEY; i++) {

            // Create white key
            var whiteKey = new Key(this.x + (keyWidth * i), this.y, keyWidth, keyHeight, whitePitch, KeyColor.WHITE);
            whiteKey.keyboard = this.keyboard;
            this.whiteKeys.push(whiteKey)

            // If key is E, skip only one half-step, otherwise skip two
            whitePitch += (i == 2) ? 1 : 2;
        }

        // Create black keys
        var blackPitch = ((this.keyboard.firstOctave + this.ix) * KEYS_PER_OCTAVE) + this.ix + 1;
        for (var i = 0; i < N_BLACK_KEY; i++) {

            // Skip the third black key
            if (i != 2) {
                var blackKey = new Key(this.x + keyWidth * (i+0.7), this.y, keyWidth/1.5, keyHeight/2, blackPitch, KeyColor.BLACK);
                blackKey.keyboard = this.keyboard;
                this.blackKeys.push(blackKey)
            }

            // If key is E, skip only one half-step, otherwise skip two
            blackPitch += (i == 2) ? 1 : 2;
        }
    }

    update(dt, isMousePressed) {
        for (var i in this.blackKeys) {
            this.blackKeys[i].update(dt, isMousePressed);
        }

        for (var i in this.whiteKeys) {
            this.whiteKeys[i].update(dt, isMousePressed);
        }
    }

    draw() {
        for (var i in this.whiteKeys) {
            this.whiteKeys[i].draw(255);
        }

        for (var i in this.blackKeys) {
            this.blackKeys[i].draw(0);
        }
    }

    playKeyWithIndex(i, duration, velocity) {
        // Check if index is a white key
        if(i == 0 || i == 2 || i == 4 || i == 5 || i == 7 || i == 9 || i == 11) {
            var keyIx = i;
            if(i < 5) {
                keyIx = i - Math.ceil(i/2);
            }
            else {
                keyIx = i - Math.floor(i/2);
            }

            this.whiteKeys[keyIx].velocity = velocity;
            this.whiteKeys[keyIx].releaseTimer = duration;
            this.whiteKeys[keyIx].pressed = true;
        }
        else {
            // 1, 3, 6, 8, 10
            var keyIx = i;
            if(i < 6) {
                keyIx -= Math.ceil(i/2);
            }
            else {
                keyIx -= Math.ceil((i+1)/2);
            }

            this.blackKeys[keyIx].velocity = velocity;
            this.blackKeys[keyIx].releaseTimer = duration;
            this.blackKeys[keyIx].pressed = true;
        }
    }

    releaseAllKeys() {
        for (var i in this.whiteKeys) {
            this.whiteKeys[i].release();
        }

        for (var i in this.blackKeys) {
            this.blackKeys[i].release();
        }
    }
}

class Keyboard {
    constructor(x, y, keyWidth, keyHeight, octaves, firstOctave) {
        this.x = x;
        this.y = y;
        this.w = N_WHITE_KEY * octaves * keyWidth;
        this.h = keyHeight;
        this.firstOctave = firstOctave;

        // Create octaves
        this.octaves = []
        for(var i = 0; i < octaves; i++) {
            var octave = new Octave(this, i, keyWidth, keyHeight);
            this.octaves.push(octave);
        }
    }

    update(dt, isMousePressed) {
        for (var i in this.octaves)
            this.octaves[i].update(dt, isMousePressed);
    }

    draw() {
        for (var i in this.octaves)
            this.octaves[i].draw();
    }

    playKeyWithPitch(pitch, duration, velocity) {
        var keyIx = pitch % 12;
        var octIx = Math.floor(pitch/12);

        if (octIx < this.octaves.length) {
            this.octaves[octIx].playKeyWithIndex(keyIx, duration, velocity);
            return true;
        }

        return false;
    }

    releaseAllKeys() {
        for (var i in this.octaves)
            this.octaves[i].releaseAllKeys();
    }

    keyPressed(keyCode) {
        switch (keyCode) {
            case 65: // A
                this.octaves[2].whiteKeys[0].pressed = true;
                break;
            case 87: // W
                this.octaves[2].blackKeys[0].pressed = true;
                break;
            case 83: // S
                this.octaves[2].whiteKeys[1].pressed = true;
                break;
            case 69: // E
                this.octaves[2].blackKeys[1].pressed = true;
                break;
            case 68: // D
                this.octaves[2].whiteKeys[2].pressed = true;
                break;
            case 70: // F
                this.octaves[2].whiteKeys[3].pressed = true;
                break;
            case 84: // T
                this.octaves[2].blackKeys[2].pressed = true;
                break;
            case 71: // G
                this.octaves[2].whiteKeys[4].pressed = true;
                break;
            case 89: // Y
                this.octaves[2].blackKeys[3].pressed = true;
                break;
            case 72: // H
                this.octaves[2].whiteKeys[5].pressed = true;
                break;
            case 85: // U
                this.octaves[2].blackKeys[4].pressed = true;
                break;
            case 74: // J
                this.octaves[2].whiteKeys[6].pressed = true;
                break;
            case 75: // k
                keyboard.octaves[3].whiteKeys[0].pressed = true;
                break;
            case 79: // O
                this.octaves[3].blackKeys[0].pressed = true;
                break;
            case 76: // L
                this.octaves[3].whiteKeys[1].pressed = true;
                break;
            case 80: // P
                this.octaves[3].blackKeys[1].pressed = true;
                break;
            case 186: // ;
                this.octaves[3].whiteKeys[2].pressed = true;
                break;
            case 222: // ;
                this.octaves[3].whiteKeys[3].pressed = true;
                break;
            case 221: // ;
                this.octaves[3].blackKeys[2].pressed = true;
                break;
            default:
                break;
        }
    }

    keyReleased(keyCode) {
        switch (keyCode) {
            case 65: // A
                this.octaves[2].whiteKeys[0].pressed = false;
                break;
            case 87: // W
                this.octaves[2].blackKeys[0].pressed = false;
                break;
            case 83: // S
                this.octaves[2].whiteKeys[1].pressed = false;
                break;
            case 69: // E
                this.octaves[2].blackKeys[1].pressed = false;
                break;
            case 68: // D
                this.octaves[2].whiteKeys[2].pressed = false;
                break;
            case 70: // F
                this.octaves[2].whiteKeys[3].pressed = false;
                break;
            case 84: // T
                this.octaves[2].blackKeys[2].pressed = false;
                break;
            case 71: // G
                this.octaves[2].whiteKeys[4].pressed = false;
                break;
            case 89: // Y
                this.octaves[2].blackKeys[3].pressed = false;
                break;
            case 72: // H
                this.octaves[2].whiteKeys[5].pressed = false;
                break;
            case 85: // U
                this.octaves[2].blackKeys[4].pressed = false;
                break;
            case 74: // J
                this.octaves[2].whiteKeys[6].pressed = false;
                break;
            case 75: // k
                this.octaves[3].whiteKeys[0].pressed = false;
                break;
            case 79: // O
                this.octaves[3].blackKeys[0].pressed = false;
                break;
            case 76: // L
                this.octaves[3].whiteKeys[1].pressed = false;
                break;
            case 80: // P
                this.octaves[3].blackKeys[1].pressed = false;
                break;
            case 186: // ;
                this.octaves[3].whiteKeys[2].pressed = false;
                break;
            case 222: // ;
                this.octaves[3].whiteKeys[3].pressed = false;
                break;
            case 221: // ;
                this.octaves[3].blackKeys[2].pressed = false;
                break;
            default:
                break;
        }
    }
}
