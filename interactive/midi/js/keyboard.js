var N_WHITE_KEY = 7;
var N_BLACK_KEY = 6;
var KEYS_PER_OCTAVE = (N_WHITE_KEY + N_BLACK_KEY - 2);

var STROKE_WEIGHT         = 2;
var RELEASED_STROKE_COLOR = 180;
var SELECTED_FILL_COLOR   = 230;
var SELECTED_STROKE_COLOR = 180;
var PLAYING_FILL_COLOR    = 100;
var PLAYING_STROKE_COLOR  = 180;

var KeyState = {
  RELEASED: 0,
  SELECTED: 1,
  PLAYING : 2,
};

class Key {
    constructor(x, y, w, h, pitch, keyboard) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
        this.pitch = pitch;
        this.state = KeyState.RELEASED;
        this.isKeyPressed = false;

        this.keyboard = keyboard;
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
            this.keyboard.keyDown(this.pitch);
            this.state = KeyState.PLAYING;
        }
    }

    release() {
        this.keyboard.keyUp(this.pitch);
        this.state = KeyState.RELEASED;
    }

    select() {
        this.state = KeyState.SELECTED;
    }

    update() {
        if(this.isMouseOver() || this.isKeyPressed) {
            if(this.keyboard.didMouseClick || this.isKeyPressed) {
                this.play();
            }
            else {
                this.select();
            }
        }
        else {
            this.release();
        }
    }

    draw(releaseFillColor) {
        strokeWeight(STROKE_WEIGHT);

        // Set fill and stroke colors
        switch (this.state) {
            case KeyState.RELEASED:
                fill(releaseFillColor);
                stroke(RELEASED_STROKE_COLOR);
                break;
            case KeyState.SELECTED:
                fill(SELECTED_FILL_COLOR);
                stroke(SELECTED_STROKE_COLOR);
                break;
            case KeyState.PLAYING:
                fill(PLAYING_FILL_COLOR);
                stroke(PLAYING_STROKE_COLOR);
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

        this.constructKeys(keyWidth, keyHeight);
    }

    constructKeys(keyWidth, keyHeight) {
        // Create white keys
        var whitePitch = ((this.keyboard.firstOctave + this.ix) * KEYS_PER_OCTAVE) + this.ix;
        for (var i = 0; i < N_WHITE_KEY; i++) {

            // Create white key
            var whiteKey = new Key(this.x + (keyWidth * i), this.y, keyWidth, keyHeight, whitePitch);
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
                var blackKey = new Key(this.x + keyWidth * (i+0.7), this.y, keyWidth/1.5, keyHeight/2, blackPitch);
                blackKey.keyboard = this.keyboard;
                this.blackKeys.push(blackKey)
            }

            // If key is E, skip only one half-step, otherwise skip two
            blackPitch += (i == 2) ? 1 : 2;
        }
    }

    update() {
        for (var i in this.whiteKeys) {
            this.whiteKeys[i].update();
        }

        for (var i in this.blackKeys) {
            this.blackKeys[i].update();
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

    releaseAllKeys() {
        for (var i in this.whiteKeys) {
            this.whiteKeys[i].state = KeyState.RELEASED;
        }

        for (var i in this.blackKeys) {
            this.blackKeys[i].state = KeyState.RELEASED;
        }
    }

    keysInState(state) {
        var keysInState = 0;
        for (var i in this.whiteKeys) {
            if(this.whiteKeys[i].state == state)
                keysInState++;
        }

        for (var i in this.blackKeys) {
            if(this.blackKeys[i].state == state)
                keysInState++;
        }

        return keysInState;
    }
}

class Keyboard {
    constructor(x, y, keyWidth, keyHeight, octaves, firstOctave) {
        this.x = x;
        this.y = y;
        this.w = N_WHITE_KEY * octaves * keyWidth;
        this.h = keyHeight;
        this.firstOctave = firstOctave;

        this.didMouseClick = false;

        this.octaves = []
        for(var i = 0; i < octaves; i++) {
            var octave = new Octave(this, i, keyWidth, keyHeight);
            this.octaves.push(octave);
        }
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

    releaseAllKeys() {
        for (var i in this.octaves)
            this.octaves[i].releaseAllKeys();
    }

    keysInState(state) {
        var keysInState = 0;
        for (var i in this.octaves)
            keysInState += this.octaves[i].keysInState(state);
        return keysInState;
    }

    update() {
        for (var i in this.octaves)
            this.octaves[i].update();
    }

    draw() {
        for (var i in this.octaves)
            this.octaves[i].draw();
    }
}
