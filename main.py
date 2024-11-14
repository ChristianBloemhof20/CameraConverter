from camera_feed import CameraFeed

def main():
    try:
        camera = CameraFeed()
        camera.getCameraFeed() 
        camera.close()
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()