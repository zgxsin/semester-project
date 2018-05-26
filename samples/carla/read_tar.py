# import tarfile
# tar = tarfile.open("/Users/zhou/Desktop/dataset/train/RGB.tar")
#
#
# tar.getmembers()
#
# import tarfile,os
# import sys
# os.chdir("/Users/zhou/Desktop/dataset/train")
# tar = tarfile.open("RGB.tar")
# for member in tar.getmembers():
#     f = tar.extractfile(member)
#     content=f.read()
#     print ("%s has %d newlines" %(member, content.count("\n")))
#     print ("%s has %d spaces" % (member,content.count(" ")))
#     print ("%s has %d characters" % (member, len(content)))
#     sys.exit()
# tar.close()


#
import os
import tarfile
directory = ["/Users/zhou/Desktop/dataset_example/train/",
        "/Users/zhou/Desktop/dataset_example/train/",
         "/Users/zhou/Desktop/dataset_example/val/",
        "/Users/zhou/Desktop/dataset_example/val/",
              ]
for i in range(len(directory)):
    if not os.path.exists(directory[i]):
        os.makedirs(directory[i])


with tarfile.open('/Users/zhou/Desktop/dataset/train/RGB.tar', 'r' ) as tar:
    tar.extractall(path=directory[0])
    tar.close()