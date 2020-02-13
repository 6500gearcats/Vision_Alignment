import cv2, random, math
#pym='LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1 python3'
#import numpy as np
#print("yoyo")
debug = False
radius = 0.9
ksize = int(6 * round(radius) + 1)
minLineLength = 1
maxLineGap = 100
red = [0, 80]
green = [80, 255]
blue = [0, 105]
imgwidth = 640
imgheight = 480
fld = cv2.ximgproc.createFastLineDetector(_length_threshold=3, _distance_threshold=2.5, _canny_th1=100, _canny_th2=100, _canny_aperture_size=7, _do_merge=True)
#part1 = cv2.GaussianBlur(src, (ksize, ksize), round(radius))
point_tolerance = 0.08
coloriters = [0xFFFFFF, 0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF00FF, 0x00FFFF, 0xFF7F7F]


def detectPoints(image):
	step1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	step2 = cv2.inRange(step1, (red[0], green[0], blue[0]), (red[1], green[1], blue[1]))
	lines = None
	lines = fld.detect(step2)
	if lines is None:
		print("lol")
		return []
	if len(lines) < 6:
	#	pass
		print("hehe")
		return []
	linestr = ""
	step3 = image
	coloriter = 0xFF0000
	maxlen = 0
	linelengths = []
	for i in range(len(lines)):
		line = lines[i][0]
		linelength = math.sqrt(math.pow((line[0] - line[2]), 2) + math.pow((line[1] - line[3]), 2))
		if linelength > maxlen:
			maxlen = linelength
		linelengths.append((i, linelength))
	linelengths.sort(key = lambda x: x[1], reverse=True)
	#print(linelengths)
	for i in range(len(lines)):
		line = lines[linelengths[i][0]][0]
		linestr += str(line)
	#	cv2.line(step3, (line[0], line[1]), (line[2], line[3]), (coloriter >> 0x10, (coloriter >> 0x8) & 0xFF, coloriter & 0xFF))
	#	coloriter = coloriters[i % 8]
	#cv2.imwrite("lines.png", step3)
	#print(len(lines), linestr)
	points = []
	if len(linelengths) >= 6:
		for i in range(len(linelengths)):
			line = lines[linelengths[i][0]][0]
			notinpointsA = True
			notinpointsB = True
			for point in points:
				if abs(line[0] - point[0]) < maxlen * point_tolerance and abs(line[1] - point[1]) < maxlen * point_tolerance:
					notinpointsA = False
				if abs(line[2] - point[0]) < maxlen * point_tolerance and abs(line[3] - point[1]) < maxlen * point_tolerance:
					notinpointsB = False
			if len(points) == 0:
				points.append((line[0], line[1]))
				points.append((line[2], line[3]))
			else:
				if notinpointsA:
					points.append((line[0], line[1]))
				if notinpointsB:
					points.append((line[2], line[3]))
	else:
		print("keke")
		return []
	if len(points) < 8:
	#	pass
		#print("pepe")
		return []
	if len(points) > 8:
		points = points[0:8]
	for x in range(len(points)):
		point = points[x]
		intx = int(round(point[0]))
		inty = int(round(point[1]))
		cv2.rectangle(step3, (intx-2, inty-2), (intx+1, inty+1), coloriters[x % 8])
		points[x] = (point[0] / imgwidth, point[1] / imgheight)
	#print(len(points), points)
	#return step3
	#print(points)
	if debug:
		cv2.imshow("frame", step3)
	return points

if __name__ == "__main__" and debug:
	cap = cv2.VideoCapture("http://10.65.0.100:1181/stream.mjpg")

	while(True):
		ret, image = cap.read()
		
		if ret:
			detectPoints(image)
			#part5 = fld.drawSegments(image, detectPoints(image))
			#print(detectPoints(image))
			#part6 = cv2.resize(part5, (640, 480))
	#		
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	#part4 = part3
	#part5 = cv2.Canny(part4,10,255,apertureSize=7)
	#lines = cv2.HoughLinesP(part5,1,np.pi/180,100,minLineLength,maxLineGap)
	#for x1,y1,x2,y2 in lines[0]:
	#    cv2.line(src,(x1,y1),(x2,y2),(0xFF,0,0),2)
	cap.release()
	cv2.destroyAllWindows()
print("hey")
