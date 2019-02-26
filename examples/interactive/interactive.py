import sdl2
import sdl2.ext

class InteractiveApp:
    def __init__(self, title="", size=(800,600), fps=30):
        # Init SDL
        sdl2.ext.init()

        # Create application window
        self.window = sdl2.ext.Window(title, size=size)

        # Create list of interactive objects
        self.interactive_objs = []

        self.fps = fps

    def start(self):
        # Starting time variables
        self.next_time = sdl2.SDL_GetTicks() + self.fps;

        # Define a variable to control the main loop
        self.is_running = True

        # Start all game objects
        for obj in self.interactive_objs:
            obj.start()

        # Show the window
        self.window.show()

        # main loop
        while self.is_running:
            events = sdl2.ext.get_events()
            self.update(events)

            self.window.refresh()

            # Search for quit events
            for event in events:
                if event.type == sdl2.SDL_QUIT:
                    self.is_running = False

    def update(self, events):
        # Update all interactive objects
        for obj in self.interactive_objs:
            obj.update(1./self.fps, events)

        time_left = self.time_left()

        sdl2.SDL_Delay(time_left)
        self.next_time += self.fps

    def time_left(self):
        now  = sdl2.SDL_GetTicks()

        if self.next_time <= now:
            return 0

        return self.next_time - now

    def add_interactive_obj(self, obj):
        self.interactive_objs.append(obj)

class InteractiveObject:
    def start(self):
        pass

    def update(self, dt, events):
        pass
