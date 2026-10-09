# import hashlib
# import os


# def hash_fichiers(chemin):
#     """Retourne un dict {chemin_relatif: hash} pour chaque fichier"""
#     resultats = {}
#     for root, dirs, files in os.walk(chemin):
#         dirs.sort()
#         for fichier in sorted(files):
#             chemin_fichier = os.path.join(root, fichier)
#             chemin_relatif = os.path.relpath(chemin_fichier, chemin)

#             hasher = hashlib.md5()
#             with open(chemin_fichier, "rb") as f:
#                 while chunk := f.read(8192):
#                     hasher.update(chunk)
#             resultats[chemin_relatif] = hasher.hexdigest()
#     return resultats


# # Comparaison détaillée
# h1 = hash_fichiers("2026-05-13-10h41")
# h2 = hash_fichiers("AR_10h41m11s_3")
# # h2 = hash_fichiers("/chemin/dossier_B")

# tous_les_fichiers = set(h1) | set(h2)
# for f in sorted(tous_les_fichiers):
#     if h1[f] == h2[f]:
#         print(f"✅ Identique : {f}")
#     if f not in h1:
#         print(f"➕ Seulement dans B : {f}")
#     elif f not in h2:
#         print(f"➖ Seulement dans A : {f}")
#     elif h1[f] != h2[f]:
#         print(f"✏️  Modifié : {f}")

"""
to test the similarity of results with the SolPOC templates.
run in your terminal: pytest result_similarity_test.py -v -s
"""

import hashlib
import os


IGNORED_FILES = {
    "Optimization.txt",
    "time.txt",
}


def hash_fichiers(chemin):
    """
    Retourne un dictionnaire :
    {chemin_relatif: hash_md5}
    pour chaque fichier du dossier.
    """
    resultats = {}

    for root, dirs, files in os.walk(chemin):
        dirs.sort()

        for fichier in sorted(files):
            chemin_fichier = os.path.join(root, fichier)
            chemin_relatif = os.path.relpath(chemin_fichier, chemin)

            hasher = hashlib.md5()

            with open(chemin_fichier, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)

            resultats[chemin_relatif] = hasher.hexdigest()

    return resultats


def compare_folders(path_a, path_b):
    """
    Compare deux dossiers récursivement.

    Returns:
        list[str]:
            Liste des différences détectées.
            Liste vide => dossiers identiques.
    """

    h1 = hash_fichiers(path_a)
    h2 = hash_fichiers(path_b)

    differences = []

    tous_les_fichiers = set(h1) | set(h2)

    for f in sorted(tous_les_fichiers):
        if f in IGNORED_FILES:
            print(f"⚠️  Ignoré : {f}")
            continue
        if f not in h1:
            differences.append(f"➕ Seulement dans B : {f}")

        elif f not in h2:
            differences.append(f"➖ Seulement dans A : {f}")

        elif h1[f] != h2[f]:
            differences.append(f"✏️  Modifié : {f}")

        else:
            print(f"✅ Identique : {f}")

    return differences


def test_template_similarity():
    """
    Vérifie que les résultats du scheduler
    sont identiques aux templates SolPOC.
    """

    path_template = "2026-05-13-10h41"
    path_scheduler = "AR_21h09m37s_3"

    assert os.path.exists(path_template), f"❌ Dossier introuvable : {path_template}"

    assert os.path.exists(path_scheduler), f"❌ Dossier introuvable : {path_scheduler}"

    differences = compare_folders(
        path_template,
        path_scheduler,
    )

    if differences:
        print("\n🚨 Différences détectées :\n")

        for diff in differences:
            print(diff)

    assert differences == [], "❌ Les dossiers comparés ne sont pas identiques."

    print("\n🎉 Tous les fichiers sont identiques !")
