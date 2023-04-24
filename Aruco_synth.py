import platform, sys, os, math, subprocess, wave
import numpy as np
import simpleaudio as sa
from struct import pack
from scipy import signal
import cv2

MIN_PHASE = -math.pi/2
MAX_PHASE =  math.pi/2

MIN_FREQ  =  110.0
MAX_FREQ  =  1760.0


"""
 resizes image to fit within a max size range
 outputs a size fitted to max_size
 inputs:
     size       - current size of the video
     max_size   - maximum allowed size
"""
def imgResize(size, max_size):
    if size[0] > max_size[0] or size[1] > max_size[1]:
        if(size[0] > size[1]):
            ratio = size[0] / size[1]
            diff = size[0] - max_size[0]
            size = (size[0] - diff, size[1] - int(diff / ratio))
        else:
            ratio = size[1] / size[0]
            diff = size[1] - max_size[1]
            size = (size[0] - int(diff / ratio), size[1] - diff)

    return size

"""
 wave genration method
 outputs data to a .wav file
 inputs:
     file       - file to write data to
     freqs      - frquencies of the waves to write
     samp_rate  - the sample rate of the file
     dur        - duration of sound to write
     vols       - volumes or amplitudes of waves
     forms      - wave form type to write (sin, sqaure, or saw)
"""
def gen_multi_wave(file, freqs, phases, samp_rate, dur, vols, forms):
    vs = vols * 100
    # initalize wave value to 0
    value = 0
    whole_wave = []
    whole_data = bytes()

    for seg in range(int(dur * samp_rate)):
        for i in range(len(freqs)):
            # basic sinasoid equation that is passes to each wave type
            equation = ((2 * math.pi) * (freqs[i] + phases[i]) * float(seg) / float(samp_rate))
            if forms[i] == "sin":
                value += int(vs[i] * np.sin(equation))
            elif forms[i] == "square":
                value += int(vs[i] * signal.square(equation))
            elif forms[i] == "saw":
                value += int(vs[i] * signal.sawtooth(equation))


        # we use pack to put the data into an acceptable format
        data = pack('<h', value)
        whole_wave.append(value)
        file.writeframesraw(data)

    num_chan = file.getnchannels()
    bps      = file.getsampwidth()
    tmp_fmt = "<"
    # here we are going to extend the wave by four times the durination 
    # in hopes that this makes the audio output during processing more smooth
    # during lag from processing
    for i in range(len(whole_wave)*4):
        whole_data += pack('<h', whole_wave[i%len(whole_wave)])


    play_obj = sa.play_buffer(whole_data, num_chan, bps, samp_rate)


"""
 converts to the frequency domain from the position of the marker
 inputs:
     min_freq       - the minimum allowed frequency
     max_freq       - the maximum allowed frequency
     intdex         - the current pixel positon given by the marker
     max_index      - the number of pixels in given direction
"""
def convert_to_freq(min_freq, max_freq, index, max_index):
    return min_freq * math.pow(max_freq/min_freq, index/float(max_index-1))

"""
 converts current pixel position data to a phase output
 this is done by taking the different of the min_phase and max_phase
 and multiplying it by the division of the index by the max_index
 and then adding from the product to the min_phase
 intput:
     min_phase      - the minimum allowed phase (should be -pi/2)
     max_phase      - the maximum allowed phase (should be pi/2)
     index          - current pixel given by marker position
     max_index      - the number of pixels in given direction
"""
def convert_to_phase(min_phase, max_phase, index, max_index):
    return min_phase + ((max_phase - min_phase) * (index / max_index))

"""
 gets the camera intrinsics and distortion matrix from
 the camera. takes in a pre-recoreded calibration video.
 video can be any length, but this method only reads the
 first 25 frames.
 inputs:
     cap        - video capture Object of calibration video
     size       - size of the video capture
"""
def getIntrinsics(cap, size):
    # needed values for calibration
    chess_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + \
                  cv2.CALIB_CB_NORMALIZE_IMAGE + \
                  cv2.CALIB_CB_FAST_CHECK

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    Bsize = (4,6)

    # object points
    objp = np.zeros((Bsize[0]*Bsize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:Bsize[0],0:Bsize[1]].T.reshape(-1,2)
    # stores object and frame points for all items
    Opoints = []
    Fpoints = []

    frame_count = 0
    # loop through frames
    while(cap.isOpened() and frame_count < 25):
        # read
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(
                    frame,
                    size,
                    fx=0,fy=0,
                    interpolation = cv2.INTER_CUBIC
                    )
            # covert to grey
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ehance greyscale
            grey_enh = ~grey
            # find corners
            ret, corners = cv2.findChessboardCorners(grey_enh, (Bsize[0], Bsize[1]), flags=chess_flags)
            if ret == True:
                Opoints.append(objp)
                # refine corners
                refCorners = cv2.cornerSubPix(grey_enh, corners, (11,11), (-1,-1), criteria)
                Fpoints.append(corners)

        frame_count += 1

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(Opoints, Fpoints, grey.shape[::-1], None, None)

    return ret, mtx, dist

def main(argv):
    # some values needed for writing sound
    sample_rate = 44100  # in Hz
    output = "sound.wav"
    # set information for output file
    audio = wave.open(output, 'wb')
    audio.setnchannels(1) # Mono output
    audio.setsampwidth(2) # sets byte size of samples
    audio.setframerate(sample_rate) # set sample rate

    # set up aruco dictionary
    ArucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    # set up aruco parameters
    ArucoParam = cv2.aruco.DetectorParameters_create()
    # NOTE: if video captue does not open try changing "/dev/video0" to 0
    cap = cv2.VideoCapture("/dev/video0")
    cali_cap = cv2.VideoCapture("calibrate.mp4")
    if not cap or not cali_cap:
        print("Error video file failed to open, please check /dev/ for video0")

    # get video size
    max_size = (1280, 720)
    size      = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cali_size = (int(cali_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                 int(cali_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # resize image
    size = imgResize(size, max_size)
    cali_size = imgResize(cali_size, max_size)
    # get frame rate 
    fps = cap.get(cv2.CAP_PROP_FPS)
    cali_fps = cali_cap.get(cv2.CAP_PROP_FPS)
    # set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mov', fourcc, fps, size, True)
    # get camera intrinsics matrix and distortion coefficents
    print("Getting camera intrinsics matrix and distortion coefficents...")
    ret, mtx, dist = getIntrinsics(cali_cap, cali_size)
    if ret:
        print("Success")
    else:
        print("Failure getting inrinsics and distortion")

    # loop through frames
    ids = []
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            # resize frame to appropriate size
            frame = cv2.resize(
                    frame,
                    size,
                    fx=0,fy=0,
                    interpolation = cv2.INTER_CUBIC
                    )
            # convert to grey
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = cv2.aruco.detectMarkers(grey,
                                                    ArucoDict,
                                                    parameters=ArucoParam
                                                    )

            if ids is not None:
                freqs = []  # frequencies
                vols  = []  # volumes (unchanging 100)
                forms = []  # wave forms (sin, square, saw)
                phz   = []  # phases
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
                # draw axis for each marker
                for i in range(ids.size):
                    cv2.drawFrameAxes(frame, mtx, dist, rvecs[i], tvecs[i], 0.05)
                    coords = (int(corners[i][0][0][0])+2, int(corners[i][0][0][1])-5)
                    msg = "X: " + str(coords[0]) + " Y: " + str(coords[1])
                    cv2.putText(frame, msg, coords, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,0,0), 2)
                    freqs.append(convert_to_freq(MIN_FREQ, MAX_FREQ, size[1] - coords[1], size[1]))
                    phz.append(convert_to_phase(MIN_PHASE, MAX_PHASE, coords[0], size[0]))
                    vols.append(200)
                    # mod id to get wave form
                    wave_type_index = ids[i] % 3

                    if wave_type_index == 0:
                        forms.append("sin")
                    elif wave_type_index == 1:
                        forms.append("square")
                    elif wave_type_index == 2:
                        forms.append("saw")
                    else:
                        forms.append("sin")
                # generate combined waveform for this frame
                gen_multi_wave(audio, freqs, phz, sample_rate, 1/fps, vols, forms)

            # show and write frame
            cv2.imshow('frame', frame)
            out.write(frame)
            # end if q key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


    # destroy windows
    cv2.destroyAllWindows()
    # release video capture
    cap.release()
    cali_cap.release()
    out.release()
    audio.close()

    # use ffmpeg to convert to mp4
    ffmpeg = "/bin/ffmpeg"
    command = [ffmpeg, "-i", "output.mov", "output.mp4"]
    if subprocess.run(command).returncode == 0:
        print("FILE saved as output.mp4")
        rm = "/bin/rm"
        command = [rm, "-f", "output.mov"]
        subprocess.run(command)
    else:
        print("Issue converting to mp4")

if __name__ == '__main__':
    main(sys.argv)
