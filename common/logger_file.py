import logging

def setup_logger(name):
    """
    Sets up a logger with the specified name.
    """
    # Create a logger with the specified name
    logger = logging.getLogger(name)

    # Set the logging level (INFO in this case)
    logger.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    return logger

def log_info(logger, message):
    """
    Logs an information level message using the given logger.
    """
    logger.info(message)

# Usage
my_logger = setup_logger('my_app_logger')
log_info(my_logger, 'This is an info message')