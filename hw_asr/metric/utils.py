import editdistance

# Don't forget to support cases when target_text == ''

def calc_cer(target_text, predicted_text) -> float:
    coef = len(target_text)
    if coef == 0:
        return 1.0
    cer = editdistance.distance(target_text, predicted_text) / coef
    return cer


def calc_wer(target_text, predicted_text) -> float:
    splitted_target_text = target_text.split(' ')
    splitted_predicted_text = predicted_text.split(' ')
    coef = len(splitted_target_text)
    if coef == 0:
        return 1.0
    wer = editdistance.distance(splitted_target_text, splitted_predicted_text) / coef
    return wer
