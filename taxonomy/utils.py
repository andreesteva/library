class Synonyms:
    """Helper class for dealing with multiple names in the taxonomy pointing to the same node.i
        For instance:
            'pigmented-lesion-benign'
            'pigmented-lesions-benign'

        We keep the convention that the root node names must be singular, and simple.
    """

    def __init__(self, primary_name):
        self._synonyms = []
        self._primary_name = primary_name.lower()


    def __repr__(self):
        return '<Synonyms: %r>' % self.primary_name


    def addSynonym(self, s):
        """Adds s to the list of synonyms.

        Args:
            s (str): the synonym
        """
        self._synonyms.append(s.lower())


    def isSynonym(self, s):
        """Returns true if s is a known synonym.

        Args:
         s (str): the potential synoynm

        Returns:
            True/False
        """
        s = s.lower()
        # Check if s is the primary name
        if s == self.primary_name:
            return True

        # Check if s is an exact copy of a known synonym
        for syn in self.synonyms:
            if s == syn:
                return True

        return False


    @property
    def primary_name(self):
        return self._primary_name

    @property
    def synonyms(self):
        return self._synonyms


class SynonymsList:
    """Maintains hard-coded list of synonyms for the taxonomy. Initialized with the 9 top categories."""

    primary_names = """
        cutaneous-lymphoma
        dermal-tumor-benign
        dermal-tumor-malignant
        epidermal-tumor-benign
        epidermal-tumor-malignant
        genodermatosis
        inflammatory
        pigmented-lesion-benign
        pigmented-lesion-malignant"""
    example_synonyms = """
        cutaneous-lymphoma-and-lymphoid-infiltrates
        benign-dermal-tumors-cysts-sinuses
        malignant-dermal-tumor
        epidermal-tumors-hamartomas-milia-and-growths-benign
        epidermal-tumors-pre-malignant-and-malignant
        genodermatoses-and-supernumerary-growths
        inflammatory
        pigmented-lesions-benign
        pigmented-lesions-malignant"""


    def __init__(self):
        """Initialize the list with the synonyms of our top 9 categories."""
        self._synonyms = []
        primaries = [p.strip() for p in self.primary_names.strip().split()]
        example_syns = [e.strip() for e in self.example_synonyms.strip().split()]
        for p, e in zip(primaries, example_syns):
            s = Synonyms(p)
            s.addSynonym(e)
            self._synonyms.append(s)


    def __repr__(self):
        return '<SynonymsList: %d entries>' % len(self._synonyms)


    def synonymOf(self, s):
        """Returns the best synonym of s, or None."""
        for syn in self.synonyms:
            if syn.isSynonym(s):
                return syn.primary_name
        return None


    @property
    def synonyms(self):
        return self._synonyms


def synset_human():
    """Returns the synset in human-readable format"""
    synset = [
            'Cutaneous lymphoma and lymphoid infiltrates',
            'Benign dermal tumors, cysts, sinuses',
            'Malignant dermal tumor',
            'Benign epidermal tumors, hamartomas, milia, and growths',
            'Malignant and premalignant epidermal tumors',
            'Genodermatoses and supernumerary growths',
            'Inflammatory Conditions',
            'Benign melanocytic lesions',
            'Malignant melanocytic lesions',
            ]
    return synset
