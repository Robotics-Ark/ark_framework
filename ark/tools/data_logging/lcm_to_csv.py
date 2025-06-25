
import struct
import pandas as pd
import argparse
from arktypes import *  # Assuming these are your generated LCM message types

LCM_SYNC_WORD = 0xEDA1DA01  # The sync word for LCM log events


class LCMLogParser:
    def __init__(self, input_filename, channel_config):
        """
        Initialize the LCM log parser.

        :param input_filename: Path to the LCM log file.
        :param channel_config: A list of tuples (channel_name, lcm_message_type) for decoding.
        """
        self.input_filename = input_filename
        self.channel_config = {channel: message_type for channel, message_type in channel_config}
        self.df = None
        
    def parse(self):
        """
        Parse the LCM log file into a Pandas DataFrame.
        :return: Pandas DataFrame containing the parsed and decoded data.
        """
        events = []

        with open(self.input_filename, 'rb') as log_file:
            event_number = 0

            while True:
                header = log_file.read(28)
                if len(header) < 28:
                    break  # End of file

                sync_word, event_number_upper, event_number_lower, timestamp_upper, timestamp_lower, channel_length, data_length = struct.unpack('>I2I2I2I', header)

                if sync_word != LCM_SYNC_WORD:
                    raise ValueError(f"Sync word mismatch. Expected {hex(LCM_SYNC_WORD)} but got {hex(sync_word)}.")

                event_number = (event_number_upper << 32) | event_number_lower
                timestamp = (timestamp_upper << 32) | timestamp_lower
                channel_name = log_file.read(channel_length).decode('utf-8')
                message_data = log_file.read(data_length)
                
                decoded_message_json = None
                if channel_name in self.channel_config:
                    try:
                        message_type = self.channel_config[channel_name]
                        decoded_message = message_type.decode(message_data)
                        decoded_message_json = self.decode_to_json(decoded_message)
                    except Exception as e:
                        print(f"Error decoding {channel_name} data: {e}")
                else:
                    print(f"Unknown channel {channel_name} (data not decoded)")

                events.append({
                    'Event Number': event_number,
                    'Timestamp': timestamp,
                    'Channel': channel_name,
                    'Data Length': data_length,
                    'Message Data': message_data.hex(),
                    'Decoded Message': decoded_message_json
                })

        return pd.DataFrame(events)
    
    @staticmethod
    def decode_to_json(decoded_message):
        """
        Convert an LCM decoded message to a JSON-like dictionary.
        :param decoded_message: The decoded LCM message.
        :return: A dictionary representing the message.
        """
        message_dict = {}
        for field in decoded_message.__slots__:
            value = getattr(decoded_message, field)
            if isinstance(value, (list, tuple)):
                message_dict[field] = [LCMLogParser.decode_to_json(v) if hasattr(v, '__slots__') else v for v in value]
            elif hasattr(value, '__slots__'):
                message_dict[field] = LCMLogParser.decode_to_json(value)
            else:
                message_dict[field] = value
        return message_dict

    def save_to_csv(self, output_filepath):
        """
        Save the parsed LCM log data to a CSV file.
        :param output_filepath: Path to save the CSV file.
        """
        if self.df is None:
            self.df = self.parse()
        self.df.to_csv(output_filepath, index=False)
        print(f"Data has been saved to {output_filepath}")
        
    def get_dataframe(self):
        if self.df is None:
            self.df = self.parse()
        return self.df
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse LCM log files and output to CSV.")
    parser.add_argument('input_filepath', help="Path to the input LCM log file")
    parser.add_argument('output_filepath', help="Path to the output CSV file")
    args = parser.parse_args()

    channel_config = [
        ("viper/joint_states", joint_state_t),
      #  ("transforms", ee_pos_t),
    ]
    
    parser = LCMLogParser(args.input_filepath, channel_config)
    parser.save_to_csv(args.output_filepath)