import sys


def get_error_details(error_msg: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename

    error_message = f"The error Occurred in Python Script {filename} at line no:{exc_tb.tb_lineno}\nAnd Error:{error_msg}"

    return error_message


class CustomException(Exception):
    def __init__(self, error: Exception, error_detail: sys):
        super().__init__(error)

        self.error_msg = get_error_details(error, error_detail)

    def __str__(self):
        return self.error_msg


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        raise CustomException(e, sys)