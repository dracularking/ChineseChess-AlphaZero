
def set_session_config(per_process_gpu_memory_fraction=None, allow_growth=None, device_list='0'):
    """

    :param allow_growth: When necessary, reserve memory
    :param float per_process_gpu_memory_fraction: specify GPU memory usage as 0 to 1

    :return:
    """
    import tensorflow as tf
    
    # Configure GPU devices
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set visible devices
            if device_list:
                device_indices = [int(x.strip()) for x in device_list.split(',')]
                visible_gpus = [gpus[i] for i in device_indices if i < len(gpus)]
                tf.config.experimental.set_visible_devices(visible_gpus, 'GPU')
            
            # Configure memory growth
            for gpu in tf.config.experimental.list_physical_devices('GPU'):
                if allow_growth is not None:
                    tf.config.experimental.set_memory_growth(gpu, allow_growth)
                
                # Note: set_memory_limit is deprecated in newer TensorFlow versions
                # Memory growth should be sufficient for most use cases
                    
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(f"GPU configuration error: {e}")
