pip

image_filename_list = ['./test_images/test5.jpg', './test_images/straight_lines2.jpg', './test_images/test6.jpg']
image_list = []


for i in range(len(image_filename_list)):
    filename = image_filename_list[i]
    original_img = cv2.imread(filename)
    result_img, binary_warped = process_image(original_img)
    image_list.append(original_img[:,:,::-1])
    image_list.append(result_img[:,:,::-1])
    image_list.append(binary_warped)

#%matplotlib inline



plt.figure(figsize=(20, 12))
for i in range(len(image_list)):
    plt.subplot(3, 3, i+1)
    plt.imshow(image_list[i])
plt.show()

print(right_fit)