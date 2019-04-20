var noteSystem = null;

var w = 20;
var h = 5;

var greenPart = {r:0,   g:120, b:80, a:255};
var redPart   = {r:160, g:0,   b:20, a:255};

var keyEffects = []

var sentimentStr = "positive";

class KeyHitEffect {
    constructor(x, y, w, h) {
      this.x = x;
      this.y = y;
      this.w = w;
      this.h = h;
      this.hitTimer = 0;
    }
}

// Jitter class
class NoteParticle {
  constructor(x, y, w, h, color) {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.color = color;
    this.colorWeight = 1.0;
    this.isAlive = false;
    this.speed = 5;
  }

  move() {
    this.x += this.speed;
  }

  kill() {
      this.x = 0;
      this.w = w;
      this.isAlive = false;
      this.colorWeight = 1.0;
  }

  draw() {
      noStroke();
      fill(this.color.r, this.color.g, this.color.b, this.color.a * this.colorWeight);
      rect(this.x, this.y, this.w, this.h);
  }
}

class NoteParticleSystem {
    constructor(size) {
        this.size = size;
        this.particles = [];

        for(var i = 0; i < this.size; i++) {
            var part = new NoteParticle(0, 0, w, h, greenPart);
            this.particles.push(part);
        }
    }

    shoot(i, noteLength, noteVelocity, noteColor) {
        for(var j = 0; j < this.particles.length; j++) {
            var part = this.particles[j];
            if(!part.isAlive) {
                part.isAlive = true;

                part.w = w * noteLength;
                part.x = -part.w;
                part.y = windowHeight - 42 - (i * h);
                part.color = noteColor;
                part.colorWeight = noteVelocity;

                keyEffects[i].hitTimer = noteLength*16*4;
                break;
            }
        }
    }

    draw() {
        for(var i = 0; i < this.particles.length; i++) {
            var part = this.particles[i];
            if(part.isAlive) {
                part.move();
                part.draw();

                if (part.x > width + part.w) {
                    part.kill();
                }
            }
        }
    }
}


// ----------------------------------------
// SETUP FUNCTIONS
// ----------------------------------------

function setup() {
    background(30, 30, 30, 220);

    // Create canvas
    canvas = createCanvas(windowWidth - 20, windowHeight - 20);
    noteSystem = new NoteParticleSystem(128);

    for(var i = 0; i < 128; i++) {
        var keyEff = new KeyHitEffect(0, windowHeight - 42 - (i * h), w, h);
        keyEffects.push(keyEff);
    }

    // Create UI
    button = createButton('PLAY');
    button.position(width/2 - 100, height/2 - 50, 200, 100);
    button.mousePressed(function() {
        // Start playing
        startAIPlayer();

        // Remove any HTML elements
        removeElements();
    });
}

// ----------------------------------------
// DRAWING FUNCTIONS
// ----------------------------------------

function draw() {
    if(!isAIPlaying)
        return;

    clear();

    var dt = window.performance.now() - canvas._pInst._lastFrameTime;

    // Draw particle system
    noteSystem.draw();

    // Draw piano keys
    for(var i = 0; i < keyEffects.length; i++) {
        if(keyEffects[i].hitTimer > 0) {
            fill(255,255,255,220);
            stroke(255);
            rect(keyEffects[i].x, keyEffects[i].y, keyEffects[i].w, keyEffects[i].h);

            keyEffects[i].hitTimer -= dt;
            if(keyEffects[i].hitTimer < 0) {
                keyEffects[i].hitTimer = 0;
            }

        }
    }

    // Draw HUD: Tempo
    fill(255);
    noStroke();

    textSize(20);
    var tempoStr = str(tempo);
    text(tempoStr, width/2 - 60, 20);

    textSize(8);
    var tempoLabel = "TEMPO (BPM)";
    text(tempoLabel, width/2 - 60, 36);

    // Draw HUD: Sentiment
    textSize(20);
    text(sentimentStr, width/2 - 60 + 64, 20);

    textSize(8);
    var sentimentLabel = "SENTIMENT";
    text(sentimentLabel, width/2 - 60 + 64, 36);
}

// ----------------------------------------
// IO FUNCTIONS
// ----------------------------------------

function mousePressed() {
    // Check if sentiment label is clicked
    var minX = width/2 - 60 + 64;
    var maxX = width/2 - 60 + 64 + 80;
    var minY = 0;
    var maxY = 36;

    if (mouseX >= minX && mouseX <= maxX && mouseY >= minY && mouseY <= maxY) {
        if(sentimentStr == "positive") {
            sentimentStr = "negative";
        }
        else if(sentimentStr == "negative") {
            sentimentStr = "positive";
        }
     }
}

function mouseReleased() {

}

function keyPressed() {

}

function keyReleased() {

}

function windowResized() {
  resizeCanvas(windowWidth - 20, windowHeight - 20);
}
