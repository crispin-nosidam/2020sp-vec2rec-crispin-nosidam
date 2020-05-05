from ..models.nlpmodels import D2VModel


# TODO: Implement Factory?
class Vec2Rec:
    job_model = D2VModel()
    res_model = D2VModel()
    train_model = D2VModel()
    all_model = D2VModel()

    @staticmethod
    def add_doc(parent_dir, file_path):
        pass

    @staticmethod
    def del_doc(parent_dir, file_glob):
        pass
