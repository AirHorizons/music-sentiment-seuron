import argparse
import sentneuron as sn

# Parse arguments
parser = argparse.ArgumentParser(description='generate_shards.py')
parser.add_argument('-path' , type=str, required=True, help="Pathfile of the midi perform.")
opt = parser.parse_args()

midi_perform_file = open(opt.path, "r")

# Create encoder to parse midi perform data
encoder = sn.dataloaders.generative.EncoderMidiPerform()

# Decode each line of the midi perform file in a separate midi file
for i, line in enumerate(midi_perform_file.read().split("\n")):
    if len(line) > 0:
        encoder.write(line, "../output/encoded_" + str(i))

midi_perform_file.close()
