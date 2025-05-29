import csv

def calculate_accuracy(filepath="dataset_v3.txt"):
    """
    Berekent de nauwkeurigheid voor MatchV1 en MatchV3 op basis van MatchReal
    in het gegeven bestand.
    """
    correct_predictions_v1 = 0
    correct_predictions_v3 = 0
    total_real_matches = 0
    total_processed_records = 0
    
    try:
        with open(filepath, 'r', newline='') as f:
            # Probeer eerst komma als scheidingsteken
            try:
                reader = csv.reader(f, delimiter=',')
                header = next(reader)
                # Controleer of de header logisch lijkt met komma's
                if len(header) < 2 and '\t' in header[0]: # Waarschijnlijk toch tabs
                    raise ValueError("Mogelijk tab-gescheiden, header niet goed geparsed met komma.")
                
                # Reset de reader als de eerste poging succesvol was en het echt een CSV is
                f.seek(0)
                reader = csv.reader(f, delimiter=',')
                header = next(reader)
                delimiter_used = ','
            except (csv.Error, ValueError) as e_csv:
                # Als komma faalt, probeer tab als scheidingsteken
                f.seek(0) # Reset file pointer
                try:
                    reader = csv.reader(f, delimiter='\t')
                    header = next(reader)
                    if len(header) < 2 : # Nog steeds niet goed, onbekend formaat
                         raise ValueError("Header niet goed geparsed met tab.")
                    delimiter_used = '\t'
                except Exception as e_tab:
                    print(f"Fout bij het lezen van de header met zowel komma als tab: {e_csv}, {e_tab}")
                    print("Zorg ervoor dat het bestand 'dataset_v3.txt' correct is geformatteerd (CSV of TSV) en een header heeft.")
                    return

            # Vind de indexen van de relevante kolommen
            try:
                # We gaan er vanuit dat de kolom met de "ground truth" iets van "MatchReal" bevat
                # en onze voorspellingen "MatchV1" en "MatchV3". We zoeken flexibel.
                real_match_col_options = ['matchreal', 'realmatch', 'actualmatch']
                v1_match_col_options = ['matchv1', 'v1match']
                v3_match_col_options = ['matchv3', 'v3match']

                idx_real_match, idx_v1_match, idx_v3_match = -1, -1, -1
                
                for i, col_name in enumerate(header):
                    col_lower = col_name.strip().lower()
                    if any(opt in col_lower for opt in real_match_col_options):
                        idx_real_match = i
                    if any(opt in col_lower for opt in v1_match_col_options):
                        idx_v1_match = i
                    if any(opt in col_lower for opt in v3_match_col_options):
                        idx_v3_match = i
                
                if idx_real_match == -1:
                    print(f"Fout: Kon 'MatchReal'-kolom niet vinden. Header: {header}")
                    return
                if idx_v1_match == -1:
                    print(f"Fout: Kon 'MatchV1'-kolom niet vinden. Header: {header}")
                    # We kunnen doorgaan zonder V1 als het niet strikt nodig is, maar voor nu stoppen we.
                    return 
                if idx_v3_match == -1:
                    print(f"Fout: Kon 'MatchV3'-kolom niet vinden. Header: {header}")
                    return

                print(f"Header gevonden met scheidingsteken '{delimiter_used}': {header}")
                print(f"Kolom 'Real Match': '{header[idx_real_match]}' (Index {idx_real_match})")
                print(f"Kolom 'V1 Match': '{header[idx_v1_match]}' (Index {idx_v1_match})")
                print(f"Kolom 'V3 Match': '{header[idx_v3_match]}' (Index {idx_v3_match})")

            except ValueError as e:
                print(f"Fout bij het vinden van kolom indexen: {e}")
                print(f"Zorg ervoor dat de header ({header}) de kolommen 'MatchReal' (of variant), 'MatchV1' (of variant) en 'MatchV3' (of variant) bevat.")
                return

            for row_num, row in enumerate(reader, 1): # Begin met tellen vanaf 1 voor dataregels
                total_processed_records += 1
                try:
                    # Converteer naar '1' of '0' voor vergelijking
                    real_match_val = '1' if row[idx_real_match].strip().lower() in ['1', 'true'] else '0'
                    v1_match_val = '1' if row[idx_v1_match].strip().lower() in ['1', 'true'] else '0'
                    v3_match_val = '1' if row[idx_v3_match].strip().lower() in ['1', 'true'] else '0'

                    if real_match_val == '1':
                        total_real_matches += 1
                        if v1_match_val == '1':
                            correct_predictions_v1 += 1
                        if v3_match_val == '1':
                            correct_predictions_v3 += 1
                except IndexError:
                    print(f"Waarschuwing: Regel {row_num + 1} heeft te weinig kolommen. Overgeslagen: {row}")
                except Exception as e:
                    print(f"Fout bij verwerken van regel {row_num + 1}: {row}. Fout: {e}. Overgeslagen.")

            if total_processed_records == 0:
                print("Geen dataregels gevonden in het bestand na de header.")
                return
                
            print(f"\nTotaal aantal verwerkte dataregels: {total_processed_records}")
            print(f"Totaal aantal werkelijke matches (MatchReal='1'): {total_real_matches}")
            
            print("\n--- Scores ---")
            if total_real_matches > 0:
                accuracy_v1 = (correct_predictions_v1 / total_real_matches) * 100
                print(f"Score V1: {correct_predictions_v1}/{total_real_matches} = {accuracy_v1:.2f}%")
                accuracy_v3 = (correct_predictions_v3 / total_real_matches) * 100
                print(f"Score V3 (huidig): {correct_predictions_v3}/{total_real_matches} = {accuracy_v3:.2f}%")
            else:
                print("Geen werkelijke matches (MatchReal='1') gevonden om scores te berekenen.")

    except FileNotFoundError:
        print(f"Fout: Het bestand '{filepath}' is niet gevonden.")
    except Exception as e:
        print(f"Een onverwachte fout is opgetreden: {e}")

if __name__ == "__main__":
    calculate_accuracy() 