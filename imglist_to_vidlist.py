base_path = '/home/bo/research/dataset/jhmdb'
img_path = base_path + '/trainlist.txt'
vid_path = base_path + '/trainlist_video.txt'
vid_list = []
with open(img_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    line = line.rstrip()
    path_split = line.split('/')
    video_name = '/'.join(path_split[:-1])
    if video_name not in vid_list:
    	vid_list.append(video_name)

with open(vid_path, 'w') as file:
	for video_name in vid_list:
		file.write(video_name + '\n')