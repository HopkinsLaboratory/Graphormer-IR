import cv2
import numpy as np
import os

from time import time

########################################################################################################################

indir = r'C:\Users\Sideshow Bob\Desktop\filtered_redo2'

show_graphs = False
save_graphs = False
save_process = False
save_ROI = False
transmission_out = False

########################################################################################################################


start = time()

try:
    outdir = os.mkdir(indir+'\\output')
except FileExistsError:
    pass # if directory already exists, do nothing
outdir = indir + '\\output\\' 

try:
    errdir = os.mkdir(indir+'\\error')
except FileExistsError:
    pass    
errdir = indir+'\\error\\'

try:
    nodatadir = os.mkdir(indir+'\\no_data\\')
except FileExistsError:
    pass    
nodatadir = indir + '\\no_data\\'

try:
    roidir = os.mkdir(indir+'\\roi\\')
except FileExistsError:
    pass    
roidir = indir + '\\roi\\'

def pre_process(image):
    base_image = image.copy()

    ## cropping image - simplifies bounding box in certain cases
    cropped_image = image[90:744, 0:1600] # This works for most cases, run script once with image[90:744, 0:1600] then on failed files with image[70:744, 0:1600]
    # cropped_image = image[70:744, 0:1600]
    
    ## process image
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    dilate = cv2.dilate(thresh, kernal, iterations=1)

    ## make bounding boxes
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

    ## select the bounding box around graph and create region of interest
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h > 200 and w > 200: # conditional to specify bbox sizes
            roi = cropped_image[y: y+h, x: x+w] 
            cv2.rectangle(cropped_image, (x, y), (x+w, y+h), (36, 255, 12), 2)
    return base_image, gray, blur, thresh, kernal, dilate, roi, image


def save_process_images(file, directory, base_image, gray, blur, thresh, kernal, dilate, roi, image):
    ## CHANGED HERE 2022-11-03
    try:
        procdir = os.mkdir(directory+'\\process\\')
    except FileExistsError:
        pass    
    procdir = indir+'\\process\\'# pass # if directory already exists, do nothing
        
    fout = file[:-4]
    fout = os.path.join(procdir, fout)
    # Currently commented out for testing purposes
    # print(fout + '_base.png')
        # cv2.imwrite(fout + '_base.png', base_image)
        # cv2.imwrite(fout + file[:-4] + '_grey.png', gray)
        # cv2.imwrite(fout + file[:-4] + '_blur.png', blur)
        # cv2.imwrite(fout + file[:-4] + '_thresh.png', thresh)
        # cv2.imwrite(fout + file[:-4] + '_kernal.png', kernal)
        # cv2.imwrite(fout + file[:-4] + '_dilate.png', dilate)
        # cv2.imwrite(fout + file[:-4] + '_roi.png', roi)
        # cv2.imwrite(fout + file[:-4] + '_bbox.png', image)

def convert_to_numpy(image):
    dim_x, dim_y = image.shape[1], image.shape[0]
    np_im = np.zeros([dim_x, dim_y])
    
    for x in range(dim_x):
        for y in range(dim_y):
            pixel = image[y,x]
            if all([rbg<100 for rbg in pixel]):
                np_im[x, y] = 1
    return np_im


def find_graph_box(im):  
    dim_x, dim_y = im.shape[1], im.shape[0]
    mid_x = dim_x // 2
    mid_y = dim_y // 2
    x_line_rows = []
    y_line_columns = []
    for row in range(0,mid_y):
        if (im[row,mid_x-20:mid_x+20] == 1).all():
            x_line_rows.append(row)
            break # only take the first line
    for row in reversed(range(mid_y,dim_y)):
        if (im[row,mid_x-20:mid_x+20] == 1).all():
            x_line_rows.append(row)
            break
    for column in range(0,mid_x):
        if (im[mid_y-20:mid_y+20,column] == 1).all():
            y_line_columns.append(column)
            break
    for column in reversed(range(mid_x,dim_x)):
        if (im[mid_y-20:mid_y+20,column] == 1).all():
            y_line_columns.append(column)
            break
            
    return ((x_line_rows[0],x_line_rows[-1]), (y_line_columns[0],y_line_columns[-1]))


N_Y_TICKS = 2
N_X_TICKS = 2


def remove_ticks(graph_im):
    dim_x, dim_y = graph_im.shape

    num_y_ticks = 0
    for y in range(dim_y):
        is_tick = True
        for i in range(N_Y_TICKS):
            if graph_im[i, y] == 0:
                is_tick = False
                break
        if is_tick:
            num_y_ticks += 1
            tick_idx = 0
            while True:
                if graph_im[tick_idx, y] == 1:
                    graph_im[tick_idx, y] = 0
                    tick_idx += 1
                else:
                    break
    num_x_ticks = 0
    for x in range(dim_x):
        is_tick = True
        for i in range(N_X_TICKS):
            if graph_im[x, -(i+1)] == 0:
                is_tick = False
                break
        if is_tick:
            num_x_ticks += 1
            tick_idx = 0
            while True:
                if graph_im[x, -(tick_idx+1)] == 1:
                    graph_im[x, -(tick_idx+1)] = 0
                    tick_idx += 1
                else:
                    break


MIN_Y, MAX_Y = 0, 100
MIN_X, MAX_X = 4000, 400
MID_X= 2000
highscale= 100
lowscale= 200
midfraction=((MID_X-MIN_X)/lowscale)/((MID_X-MIN_X)/lowscale+(MAX_X-MID_X)/highscale)


def parse_graph(graph_im, is_trans):
    dim_x, dim_y = graph_im.shape

    data = []
    
    for x in range(dim_x):
        ys = []
        if x < (midfraction*dim_x):
            label_x=(float(x)/(midfraction*dim_x)*(MID_X-MIN_X)+MIN_X)
        else:
            label_x=((float(x)-(midfraction*dim_x))/(dim_x-midfraction*dim_x)*(MAX_X-MID_X)+MID_X)
        data.append((label_x,np.nan))
        for y in range(dim_y):
            if graph_im[x, y] == 1:
                label_y = (dim_y - float(y)) / dim_y * MAX_Y
                ys.append(label_y)
        if not(ys == []): 
            temp = data.pop(-1)
            temp = list(temp)
            temp[1] = np.average(ys)
            temp = tuple(temp)
            data.append(temp)
    
    maximum = np.nanmax(data, axis=0)[1]
    minimum = np.nanmin(data, axis=0)[1]
    
    if is_trans:
    ## transmission
        data[:] = [(i[0], ((minimum - i[1]) / (minimum - maximum))) for i in data]
    else:
    ## absorption 
        data[:] = [(i[0], 1 - ((minimum - i[1]) / (minimum - maximum))) for i in data]
    
    # print(time() - start)
    return data


def write_one_spectrum(file,input_directory,output_directory,error_directory):
    try: 
        fname =  os.path.join(input_directory, file)

        image = cv2.imread(fname)

        base_image, gray, blur, thresh, kernal, dilate, roi, image = pre_process(image)

        if save_process:
            save_process_images(file, input_directory, base_image, gray, blur, thresh, kernal, dilate, roi, image)

        if save_ROI:
            fout = file[:-4]

            fout = os.path.join(roidir, fout)
            cv2.imwrite(fout + file[:-4] + '_roi.png', roi)

        np_im = convert_to_numpy(roi)

        x_box, y_box = find_graph_box(np_im)
        
        graph_im = np_im[x_box[0]+1:x_box[1], y_box[0]+1:y_box[1]]

        remove_ticks(graph_im)

        data = parse_graph(graph_im, transmission_out)

        fout1 = file[:-4] + ".csv"
        fout1 = os.path.join(output_directory, fout1)
        
        np.savetxt(fout1, data, delimiter = ",")

        print(time() - start)
        print(file)
    except UnboundLocalError:
        print(time() - start)
        print("NO DATA: {0}".format(file))
        fout2 = os.path.join(nodatadir, file)
        cv2.imwrite(fout2, image)
    except Exception:
        print(time() - start)
        print("ERROR: {0}".format(file)) 
        fout2 = os.path.join(error_directory, file) 
        cv2.imwrite(fout2, base_image)

for subdir, dirs, files in os.walk(indir): # iterating through all the files in the root directory
        for file in files:
            if True:
                write_one_spectrum(file,indir,outdir,errdir)
        break
