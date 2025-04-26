import mmap


def modify_zip(path_file):
    print("Modifying file: {}".format(path_file))
    signature = b'\x50\x4b\x06\x07'

    with open(path_file, 'r+b') as f:
        # Memory-map the file (0 means whole file)
        mm = mmap.mmap(f.fileno(), 0)

        pos = mm.rfind(signature)
        if pos == -1:
            print("Signature not found in file.")
            mm.close()
            return

        # Calculate the target offset (4 bytes for signature + 12 bytes offset = 16 bytes after pos)
        target_offset = pos + 16
        current_byte = mm[target_offset]
        if current_byte != 0:
            print("Warning: The target byte at offset {} is not 0x00 as expected.".format(target_offset))
        else:
            print("Modifying byte at offset {} (from 0x00 to 0x01).".format(target_offset))

        # Modify the byte in-place
        mm[target_offset] = 1  # 1 is equivalent to 0x01
        mm.flush()  # Flush changes to disk
        mm.close()

    print("File modified successfully.")


if __name__ == '__main__':
    pth = "/data/ATM/colombia/flight_2.zip"
    modify_zip(pth)
