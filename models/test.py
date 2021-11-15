import Hierarchical

nets_arch = {

}

structure = {
    "phylum": {
        "class_Arthropoda": ["family_Arachnida", "family_Insecta"],
        "class_Chordata": ["family_Actinopterygii", "family_Amphibia", "class_Aves", "class_Mammalia", "class_Reptilia"]
        }
}

H = Hierarchical.HierarchicalClassification({}, structure)

H.predict(1)
