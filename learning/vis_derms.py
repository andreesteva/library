import numpy as np

def people_SS(test_name):
    """Returns the SS of anyone that has taken one of our binary tests.

    For details, see /media/esteva/ExtraDrive1/ThrunResearch/web/image_classifier
    It contains the tests given (pigmented, epidermal, general, etc.) and the detailed results.
    This function does not call or reference anything contained there. The results are hardcoded.

    Args:
        test_name (str): The test name.

    Returns:
        The sensitivity and specificy of all test takers as a list of lists where each entry
            has lenth 2: the first element is sensitivity, the second is specificity.
    """
    if test_name.lower() == 'pigmented_lesions_100':
        results = [
                [0.92, 0.66],   # Rob Novoa
                [0.9, 0.74],    # Justin Ko
                [0.78, 0.84],   # Brett Kuprel
                [0.74, 0.78],   # Andre Esteva
                [0.86, 0.62],   # Kerri Rieger
                [0.86, 0.7],    # Kavita Sarin
                [0.9, 0.76],    # Susan Swetter
                ]

    # 1 vs All for melanoma
    if test_name.lower() == '9way_180_melanoma':
        results = [
                [0.9, 0.4875],
                [0.85, 0.5125],
                ]

    if test_name.lower() == '9way_180_carcinomas':
        results = [
                [0.8, 0.5],
                [0.75, 0.525],
                ]

    if test_name.lower() == 'pigmented_lesions_200':
        results = [
                [0.84, 0.75], # Rob Novoa, attempt 1
                [0.5682, 0.80], # Brett Kuprel
                ]

    if test_name.lower() == 'epidermal_lesions_200':
        results = [
                [0.94, 0.87], # Rob Novoa, attempt 1
                [0.92, 0.89], # Justin Ko Malignant/Benign
                ]

    # Dermquest Tests
    if test_name.lower() == 'dermquest_pigmented':
        results = [
                [0.89, .87], # Rob Novoa, attempt 1
                ]

    if test_name.lower() == 'dermquest_pigmented_action':
        results = [
                [0.9464, .6667], # Rob Novoa, attempt 1
                ]

    if test_name.lower() == 'dermquest_epidermal':
        results = [
                [0.9367, .7183], # Rob Novoa, attempt 1
                [0.962, 0.7465], # Justin Ko
                ]

    if test_name.lower() == 'dermquest_epidermal_action': # this is biopsy vs reassure
        results = [
                [0.9494, .6056], # Rob Novoa, attempt 1
                [0.962, 0.7324], # Justin KO
                ]

    # Edinburgh Tests:
    if test_name.lower() == 'edinburgh_pigmented':
        results = [
                [0.92, .7733], # Rob Novoa, attempt 1
                ]
    if test_name.lower() == 'edinburgh_pigmented_action':
        results = [
                [0.96, .6933], # Rob Novoa, attempt 1
                ]

    # FINALIZED TESTS ----------------------

    # Epidermal
    if test_name.lower() == 'edinburgh_epidermal':
        results = [
                # [0.9733, .7867], # Rob Novoa, attempt 1
                [0.8923, 0.8143], # Sumaira
                [0.9385, 0.7429], # Jennifer
                [0.8462, 0.7857], # David
                [0.9538, 0.7143], # Teresa
                [0.9692, 0.6857], # Michelle
                [0.8769, 0.8143], # Golara
                [0.9692, 0.8571], # Tyler
                [0.8923, 0.7286], # Bernice
                [0.9692, 0.8143], # Carolyn
                [0.9385, 0.9143], # Matthew
                [0.9846, 0.7571], # Ann
                [0.9846, 0.8286], # Kristin
                [0.8923, 0.9000], # Anthony
                [0.8154, 0.9286], # Marylanne
                [0.9692, 0.7143], # Silvina
                [0.9385, 0.6286], # Kerri
                [0.9385, 0.7429], # Kavita
                [1.0000, 0.6714], # Laurel
                [0.8923, 0.7714], # Susan
                [0.9846, 0.4857], # Joyce
                [0.9077, 0.8000], # Kevin
                [0.8923, 0.9286], # Stephen
                [0.9077, 0.8000], # Jennifer2
                [0.8923, 0.8000], # Kari
                [0.9846, 0.8429], # Misha
                ]

    if test_name.lower() == 'edinburgh_epidermal_action':
        results = [
                # [0.9867, .7067], # Rob Novoa, attempt 1
                [0.9077, 0.8143], # Sumaira
                [0.9385, 0.7429], # Jennifer
                [0.9231, 0.5857], # David
                [0.9846, 0.5571], # Teresa
                [0.9846, 0.6143], # Michelle
                [0.9077, 0.7286], # Golara
                [0.9846, 0.7143], # Tyler
                [0.9692, 0.5286], # Bernice
                [0.9692, 0.8143], # Carolyn
                [0.9231, 0.9143], # Matthew
                [0.9846, 0.7571], # Ann
                [1.0000, 0.6429], # Kristin
                [0.9077, 0.8857], # Anthony
                [0.8154, 0.9000], # Marylanne
                [0.9692, 0.7143], # Silvina
                [0.9538, 0.5429], # Kerri
                [0.9538, 0.6143], # Kavita
                [1.0000, 0.6429], # Laurel
                [0.9077, 0.7714], # Susan
                [0.9846, 0.4857], # Joyce
                [0.9077, 0.8000], # Kevin
                [0.9846, 0.8000], # Stephen
                [0.9538, 0.7143], # Jennifer2
                [0.9077, 0.8000], # Kari
                [1.0000, 0.7429], # Misha
                ]

    # Melanocytic: Edinburgh+EPIC:
    if test_name.lower() == 'edinburgh_epic_pigmented':
        results = [
                # [0.9783, 0.8163], # robnovoa
                [0.9130, 0.9032], # Sumaira
                [0.9348, 0.9140], # Jennifer
                [0.8261, 0.9032], # David
                [0.9783, 0.5376], # Teresa
                [0.9783, 0.7419], # Michelle
                [0.9565, 0.8925], # Golara
                [0.9565, 0.8602], # Tyler
                [0.8261, 0.9032], # Bernice
                [0.9348, 0.8280], # Carolyn
                [0.9348, 0.8817], # Kristin
                [0.7174, 0.9570], # Anthony
                [0.8478, 0.9677], # Marylanne
                [0.9783, 0.8280], # Silvina
                [0.9783, 0.8387], # Kerri
                [0.7391, 0.9032], # Kavita
                [0.9783, 0.5591], # Laurel
                [0.9348, 0.8280], # Jennifer2
                [0.9565, 0.8817], # Susan
                [0.9783, 0.6882], # Joyce
                [0.7826, 0.9570], # Stephen
                [0.8913, 0.8710], # Kari
                [0.8913, 0.8817], # Misha
                ]

    if test_name.lower() == 'edinburgh_epic_pigmented_action':
        results = [
                # [1.0000, 0.7143], # robnovoa
                [0.9130, 0.8710], # Sumaira
                [0.9348, 0.9140], # Jennifer
                [0.9783, 0.6022], # David
                [0.9783, 0.4946], # Teresa
                [0.9783, 0.7097], # Michelle
                [0.9565, 0.8710], # Golara
                [1.0000, 0.4516], # Tyler
                [0.9565, 0.5161], # Bernice
                [0.9565, 0.7742], # Carolyn
                [0.9565, 0.7849], # Kristin
                [0.7174, 0.9570], # Anthony
                [0.8478, 0.9677], # Marylanne
                [0.9783, 0.8387], # Silvina
                [0.9783, 0.7742], # Kerri
                [0.9565, 0.7634], # Kavita
                [0.9783, 0.5591], # Laurel
                [0.9783, 0.7097], # Jennifer2
                [0.9565, 0.8495], # Susan
                [0.9783, 0.6774], # Joyce
                [0.9783, 0.8602], # Stephen
                [0.9348, 0.8172], # Kari
                [0.9565, 0.8172], # Misha
                ]

    # Dermoscopy
    if test_name.lower() == 'dermoscopy':
        results = [
                # [0.9155, 0.7250], # robnovoa
                [0.8732, 0.8000], # Sumaira
                [0.9437, 0.6750], # Jennifer
                [0.5211, 0.9500], # David
                [0.9718, 0.7250], # Teresa
                [0.9859, 0.6500], # Michelle
                [0.8451, 0.6750], # Golara
                [0.8028, 0.7250], # Tyler
                [0.8310, 0.7750], # Bernice
                [0.7183, 0.8000], # Carolyn
                [0.9014, 0.5750], # Kristin
                [0.3662, 0.9250], # Anthony
                [0.7746, 0.9000], # Marylanne
                [0.8028, 0.8250], # Kerri
                [0.5493, 0.9750], # Kavita
                [0.9437, 0.5250], # Laurel
                [0.9014, 0.6750], # Jennifer2
                [0.8732, 0.7500], # Susan
                [0.7606, 0.8500], # Joyce
                [0.9155, 0.7250], # Stephen
                [0.8310, 0.8500], # Kari
                [0.7465, 0.7750], # Misha
                ]

    if test_name.lower() == 'dermoscopy_action':
        results = [
                # [0.9155, 0.6250], # robnovoa
                [0.8732, 0.8000], # Sumaira
                [0.9577, 0.6750], # Jennifer
                [0.7042, 0.8000], # David
                [0.9718, 0.7250], # Teresa
                [0.9859, 0.6500], # Michelle
                [0.8592, 0.6250], # Golara
                [0.9859, 0.4500], # Tyler
                [0.9577, 0.5000], # Bernice
                [0.7183, 0.8000], # Carolyn
                [0.9014, 0.5750], # Kristin
                [0.3803, 0.9250], # Anthony
                [0.7746, 0.9000], # Marylanne
                [0.8028, 0.8250], # Kerri
                [0.8310, 0.7750], # Kavita
                [0.9437, 0.5250], # Laurel
                [0.9155, 0.6750], # Jennifer2
                [0.9155, 0.7250], # Susan
                [0.7606, 0.8500], # Joyce
                [0.9155, 0.6500], # Stephen
                [0.8451, 0.8500], # Kari
                [0.8732, 0.6500], # Misha
                ]

    assert len(results) >= 1, 'test does not exist or has not been completed by anyone'
    return np.array(results)


def JustinKoAnswers(ordering='skindata3'):
    """Returns Justin's Answers, together with the ground truth for them.

    To recalculate Justin's 1 vs all sensitivity and specificity, run the following code:
    answers, truth = JustinKoAnswers()
    answers = np.array(answers)
    truth = np.array(truth)
    malignant = 8
    sens = 1.0 * np.sum( answers[truth == malignant] == truth[truth == malignant] ) / np.sum(truth == malignant)
    spec = 1.0 * np.sum( answers[truth != malignant] == truth[truth != malignant] ) / np.sum(truth != malignant)
    print sens, spec

    It should print 0.9, 0.4875. For Rob it is 0.85, 0.5125

    """

    answers = '5 0 4 6 1 6 5 0 6 4 4 4 1 4 8 1 8 6 6 6 4 4 1 8 4 4 1 0 4 1 3 6 5 8 6 5 6 5 8 4 1 5 4 4 8 6 4 6 1 4 4 8 4 5 3 3 4 0 2 1 8 1 4 1 8 8 0 5 5 4 4 4 5 3 7 4 5 5 8 4 2 8 1 7 5 2 4 1 4 3 4 6 4 5 8 1 4 8 1 3 0 4 8 6 1 8 8 4 0 5 8 4 5 6 1 0 3 5 8 6 1 5 5 4 0 0 5 8 6 4 8 6 8 5 1 4 4 4 1 1 4 1 5 6 3 3 1 4 8 4 0 1 8 5 6 4 4 2 1 2 1 1 4 0 4 1 5 6 4 6 3 3 2 4 8 1 1 1 6 5'
    truth =   '2 0 1 6 3 7 2 0 1 0 0 4 8 3 3 2 8 6 7 7 0 4 1 7 3 4 4 0 8 1 3 6 2 7 6 6 8 2 8 6 2 5 0 2 3 6 0 7 4 4 8 7 4 5 4 3 7 0 2 1 0 2 7 1 8 7 0 5 5 6 4 4 5 6 6 8 5 2 8 0 1 6 7 7 5 2 4 1 8 3 4 6 7 5 8 1 3 8 1 3 0 4 8 8 1 7 1 4 0 5 7 4 5 6 1 0 3 0 8 6 1 5 5 3 0 3 5 7 6 4 8 6 8 5 2 4 4 3 7 8 2 5 5 5 6 0 1 4 8 3 4 8 2 5 6 3 7 2 6 2 1 2 3 0 3 1 5 7 2 6 3 3 2 0 7 1 1 1 2 5'
    labels= """
    0 cutaneous-lymphoma-and-lymphoid-infiltrates
    1 epidermal-tumors-hamartomas-milia-and-growths-benign
    2 pigmented-lesions-benign
    3 genodermatoses-and-supernumerary-growths
    4 inflammatory
    5 pigmented-lesions-malignant
    6 epidermal-tumors-pre-malignant-and-malignant
    7 malignant-dermal-tumor
    8 benign-dermal-tumors-cysts-sinuses
    """
    labels = labels.strip().split('\n')

    if ordering == 'skindata3':
        reorder = {0:0, 1:4, 2:7, 3:5, 4:6, 5:8, 6:3, 7:2, 8:1}
        a = [reorder[int(aa)] for aa in answers.strip().split()]
        t = [reorder[int(tt)] for tt in truth.strip().split()]
        return a, t

    raise ValueError('ordering %s not implemented' % ordering)


def RobNovoaAnswers(ordering='skindata3'):
    """Returns Rob's Answers, together with the ground truth for them."""

    answers = '2 0 8 6 3 6 5 4 6 4 0 4 8 4 3 2 8 6 7 4 4 4 1 8 4 4 4 0 4 1 4 6 2 8 6 5 7 5 8 4 5 5 4 4 4 6 4 7 8 4 7 7 4 6 4 4 4 4 2 1 4 8 4 1 8 8 7 5 5 4 4 7 5 4 6 4 7 5 1 6 2 6 1 4 5 2 4 1 4 4 4 6 2 5 6 1 4 8 1 3 0 8 8 8 1 8 8 4 4 5 8 8 5 6 1 0 3 7 8 4 1 5 5 4 0 4 5 8 6 4 8 6 8 5 1 4 4 4 1 1 2 5 5 6 1 0 4 4 8 4 4 8 2 5 6 4 8 2 6 2 1 5 4 0 4 1 5 5 3 6 4 4 2 4 4 1 1 1 7 5'
    truth =   '2 0 1 6 3 7 2 0 1 0 0 4 8 3 3 2 8 6 7 7 0 4 1 7 3 4 4 0 8 1 3 6 2 7 6 6 8 2 8 6 2 5 0 2 3 6 0 7 4 4 8 7 4 5 4 3 7 0 2 1 0 2 7 1 8 7 0 5 5 6 4 4 5 6 6 8 5 2 8 0 1 6 7 7 5 2 4 1 8 3 4 6 7 5 8 1 3 8 1 3 0 4 8 8 1 7 1 4 0 5 7 4 5 6 1 0 3 0 8 6 1 5 5 3 0 3 5 7 6 4 8 6 8 5 2 4 4 3 7 8 2 5 5 5 6 0 1 4 8 3 4 8 2 5 6 3 7 2 6 2 1 2 3 0 3 1 5 7 2 6 3 3 2 0 7 1 1 1 2 5'
    labels= """
    0 cutaneous-lymphoma-and-lymphoid-infiltrates
    1 epidermal-tumors-hamartomas-milia-and-growths-benign
    2 pigmented-lesions-benign
    3 genodermatoses-and-supernumerary-growths
    4 inflammatory
    5 pigmented-lesions-malignant
    6 epidermal-tumors-pre-malignant-and-malignant
    7 malignant-dermal-tumor
    8 benign-dermal-tumors-cysts-sinuses
    """
    labels = labels.strip().split('\n')

    if ordering == 'skindata3':
        reorder = {0:0, 1:4, 2:7, 3:5, 4:6, 5:8, 6:3, 7:2, 8:1}
        a = [reorder[int(aa)] for aa in answers.strip().split()]
        t = [reorder[int(tt)] for tt in truth.strip().split()]
        return a, t

    raise ValueError('ordering %s not implemented' % ordering)


def JustinKoAccuracy(ordering='skindata2'):
    # Justin's Top1 Accuracy: 53.3333333333 %
    # Justin's Top2 Accuracy: 68.8888888889 %
    top1 = 0.5333
    top2 = 0.6888

    if ordering == 'skindata2':
        # Accuracy per class:
        justin = """
        50.0 % 0 cutaneous-lymphoma
        55.0 % 1 dermal-tumor-benign
        5.0 % 2 dermal-tumor-malignant
        60.0 % 3 epidermal-tumor-malignant
        80.0 % 4 epidermal-tumor-benign
        35.0 % 5 genodermatoses
        80.0 % 6 inflammatory
        25.0 % 7 pigmented-lesion-benign
        90.0 % 8 pigmented-lesion-malignant
        """
    elif ordering == 'skindata2-6classes':
        # Accuracy per class:
        top1 = 0.644
        top2 = 0.805
        justin = """
        50.0 % 0 cutaneous-lymphoma
        82.5 % 1 inflammatory-and-genetic-conditions
        77.5 % 2 non-pigmented-tumor-benign
        47.5 % 3 non-pigmented-tumor-malignant
        25.0 % 4 pigmented-lesion-benign
        90.0 % 5 pigmented-lesion-malignant
        """
    else:
        # Accuracy per class:
        justin = """
        50.0 % 0 cutaneous-lymphoma-and-lymphoid-infiltrates
        80.0 % 1 epidermal-tumors-hamartomas-milia-and-growths-benign
        25.0 % 2 pigmented-lesions-benign
        35.0 % 3 genodermatoses-and-supernumerary-growths
        80.0 % 4 inflammatory
        90.0 % 5 pigmented-lesions-malignant
        60.0 % 6 epidermal-tumors-pre-malignant-and-malignant
        5.0 % 7 malignant-dermal-tumor
        55.0 % 8 benign-dermal-tumors-cysts-sinuses
        """

    justin = justin.strip().split('\n')
    labs = [j.split()[-1].split()[0] for j in justin]
    ind = np.argsort(labs)
    labs = np.array(labs)[ind]
    accs = np.array([float(j.split()[0]) / 100.0 for j in justin])[ind]

    return accs, labs, top1, top2

def RobNovoaAccuracy(ordering='skindata2'):
#     Rob's Top1 Accuracy: 55.0 %
#     Rob's Top2 Accuracy: 73.8888888889 %
    top1 = 0.55
    top2 = 0.7388

    if ordering == 'skindata2':
        rob = """
        40.0 % 0 cutaneous-lymphoma
        60.0 % 1 dermal-tumor-benign
        15.0 % 2 dermal-tumor-malignant
        70.0 % 3 epidermal-tumor-malignant
        75.0 % 4 epidermal-tumor-benign
        20.0 % 5 genodermatoses
        80.0 % 6 inflammatory
        50.0 % 7 pigmented-lesion-benign
        85.0 % 8 pigmented-lesion-malignant
        """
    elif ordering == 'skindata2-6classes':
        # Accuracy per class:
        top1 = 0.666
        top2 = 0.805
        rob= """
        40.0 % 0 cutaneous-lymphoma
        90.0 % 1 inflammatory-and-genetic-conditions
        77.5 % 2 non-pigmented-tumor-benign
        45.0 % 3 non-pigmented-tumor-malignant
        50.0 % 4 pigmented-lesion-benign
        85.0 % 5 pigmented-lesion-malignant
        """
    else:
        rob = """
        40.0 % 0 cutaneous-lymphoma-and-lymphoid-infiltrates
        75.0 % 1 epidermal-tumors-hamartomas-milia-and-growths-benign
        50.0 % 2 pigmented-lesions-benign
        20.0 % 3 genodermatoses-and-supernumerary-growths
        80.0 % 4 inflammatory
        85.0 % 5 pigmented-lesions-malignant
        70.0 % 6 epidermal-tumors-pre-malignant-and-malignant
        15.0 % 7 malignant-dermal-tumor
        60.0 % 8 benign-dermal-tumors-cysts-sinuses
        """
    rob = rob.strip().split('\n')
    labs = [j.split()[-1].split()[0] for j in rob]
    ind = np.argsort(labs)
    labs = np.array(labs)[ind]
    accs = np.array([float(j.split()[0]) / 100.0 for j in rob])[ind]

    return accs, labs, top1, top2

