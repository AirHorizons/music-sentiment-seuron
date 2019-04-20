// Sampler constants
var MAX_VELOCITY = 128;

var baseTimeDiv  = "16n";
var sequenceInit = "t_68";

var geIx = 0;
var scIx = 0;
var evIx = 0;
var score = new Array(16);

var duration    = "16n";
var noteLength  = 1;
var velocity    = 1.0;
var tempo       = 120;
var isAIPlaying = false;

function startAIPlayer() {
    // Start MAIN app
    isAIPlaying = true;

    // Make ure tone.js is running
    if (Tone.context.state !== 'running') {
        Tone.context.resume();
    }

    // Start the tone.js scheduler
    Tone.Transport.bpm.value = tempo;
    Tone.Transport.start();

    var socket = io.connect('http://' + document.domain + ':' + location.port);
    socket.on('connect', function() {
        socket.emit('generate', {init: sequenceInit, first: true, sentiment: sentimentStr});
    });

    socket.on("notes", function(sample, notes, sentiment) {
        score[geIx] = {notes: notes.split(" "), sentiment: sentiment}
        geIx = (geIx + 1) % score.length;

        // sample = sample.split(" ");
        // var sequenceInit = sample[sample.length - 1]
        socket.emit('generate', {init: sample, first: false, sentiment: sentimentStr});
    })

    Tone.Transport.scheduleRepeat(function(time) {
        if(score[scIx] == null) {
            console.log("SCORE IS OVER!");
            return
        }

        if(evIx >= score[scIx].notes.length) {
            evIx = 0;

            score[scIx].notes = [];
            score[scIx].sentiment = "";

            scIx = (scIx + 1) % score.length;
        }

        if(score[scIx].notes[evIx] != null && score[scIx].notes[evIx] == ".") {
            evIx++;
        }

        while(score[scIx].notes[evIx] != null && score[scIx].notes[evIx] != ".") {
            var ev = score[scIx].notes[evIx].split("_");

            switch (ev[0]) {
              case "n":
                var pitch = parseInt(ev[1]);
                var note = Tone.Frequency(pitch, "midi").toNote();
                sampler.triggerAttackRelease(note, duration, Tone.Transport.now(), velocity);

                var noteColor = greenPart;
                if (score[scIx].sentiment == "negative") {
                    noteColor = redPart;
                }

                noteSystem.shoot(pitch, noteLength, velocity, noteColor);
                break;
              case "d":
                // Add dots to note
                dots = "";
                for (var i = 0; i < Math.min(1, parseInt(ev[2])); i++) {
                    dots = dots + ".";
                }

                if(ev[1] != "0") {
                    noteLength = 16/parseInt(ev[1]);
                    duration = ev[1] + "n" + dots;
                }

                break;
              case "v":
                velocity = parseInt(ev[1])/MAX_VELOCITY;
                break;
              case "t":
                if(parseInt(ev[1]) >= 20) {
                    tempo = parseInt(ev[1]);
                    Tone.Transport.bpm.value = tempo;
                }
                break;
            }

            evIx++;
        }
    }, baseTimeDiv);
}

var sampler = new Tone.Sampler({
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
