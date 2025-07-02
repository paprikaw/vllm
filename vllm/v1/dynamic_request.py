from vllm.v1.request import Request

class DynamicRequest(Request):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_cur_request = True

    def set_is_cur_request(self, is_cur_request: bool):
        self.is_cur_request = is_cur_request