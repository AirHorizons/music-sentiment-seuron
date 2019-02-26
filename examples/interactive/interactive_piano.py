import sdl2
import sdl2.ext
import music21
import mido

from interactive import InteractiveObject

class InteractivePiano(InteractiveObject):
    def __init__(self, sampling_callback=None):
        self.sampling_callback = sampling_callback

        self.tempo         = 120
        self.quarterLenght = 60/self.tempo

        self.port = mido.open_output('IAC Driver IAC Bus 1')

        self.keys  = {
            sdl2.SDLK_z: 60,
            sdl2.SDLK_s: 61,
            sdl2.SDLK_x: 62,
            sdl2.SDLK_d: 63,
            sdl2.SDLK_c: 64,
            sdl2.SDLK_f: 65,
            sdl2.SDLK_v: 66,
            sdl2.SDLK_g: 67,
            sdl2.SDLK_b: 68,
            sdl2.SDLK_h: 69,
            sdl2.SDLK_n: 70,
            sdl2.SDLK_j: 71,
        }

    def start(self):
        self.sampling_counter = 0
        self.beat_counter = 0
        self.state = 0

        self.notes = []

        self.pressed = {}
        for key in self.keys:
            self.pressed[key] = 0.

    def update(self, dt, events):
        if self.state == 0:
            self.read_keys(dt, events)

            self.sampling_counter += dt
            if self.sampling_counter > 5:
                print(self.notes)
                self.state = 1
                self.sampling_counter = 0
                self.sampling_callback(self.notes)

            # Count beats
            self.beat_counter += dt
            if self.beat_counter >= self.quarterLenght:
                self.beat_counter = 0
                self.notes.append(".")

    def play_note(self, note, velocity=30):
        msg = mido.Message('note_on', note=note)
        self.port.send(msg)

    def release_note(self, note):
        msg = mido.Message('note_off', note=note)
        self.port.send(msg)

    def read_keys(self, dt, events):
        for event in events:
            if event.type == sdl2.SDL_KEYDOWN:
                key_code = event.key.keysym.sym
                if key_code in self.keys and self.pressed[key_code] == 0:
                    self.pressed[key_code] = sdl2.SDL_GetTicks()

                    pitch = self.keys[key_code]
                    self.play_note(pitch)

                    # User is still playing the seed
                    self.sampling_counter = 0

            if event.type == sdl2.SDL_KEYUP:
                key_code = event.key.keysym.sym
                if key_code in self.keys and self.pressed[key_code] > 0:
                    # Calculate note duration in seconds
                    duration_s = (sdl2.SDL_GetTicks() - self.pressed[key_code]) / 1000.

                    # Calculate note duration in quarter note lenght
                    duration_quarterLength = duration_s/self.quarterLenght

                    # Calculate note duration type (e.g whole, half, quarter, etc)
                    duration_type = music21.duration.quarterLengthToClosestType(duration_quarterLength)[0]

                    self.pressed[key_code] = 0.

                    # Get note pitch
                    pitch = self.keys[key_code]
                    self.release_note(pitch)

                    # Calculate note velocity
                    velocity = 80

                    print("n", pitch, duration_type, velocity)

                    self.notes.append("n_" + str(pitch) + "_" + duration_type + "_" + str(velocity))
