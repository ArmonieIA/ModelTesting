def tokenize_rpgiii_code(code_line):
    # Determine the starting position based on the "type" field and comment
    type_field, commented = code_line[5:6].strip(), code_line[6:7].strip()

    start_pos = 6 
    #if len(type_field) == 1 else 7 Future opti ?
    
    # Initialize tokens dictionary
    tokens = {"type": type_field}

    # Determine Specs and Define tokens based on the type field
    if type_field == "H" and commented != "*":
        # Define tokens for H-spec
        tokens.update({
            "debug": code_line[start_pos+8:start_pos+9].strip(),
            "option_c": code_line[start_pos+11:start_pos+12].strip(),
            "option_d": code_line[start_pos+12:start_pos+13].strip(),
            "option_y": code_line[start_pos+13:start_pos+14].strip(),
            "option_n": code_line[start_pos+14:start_pos+15].strip(),
            "date_edition": code_line[start_pos+34:start_pos+35].strip(),
            "file_translation": code_line[start_pos+36:start_pos+37].strip(),
            "transparent_option": code_line[start_pos+50:start_pos+51].strip(),
            "program_id": code_line[start_pos+67:].strip(),
            "comments": code_line[start_pos+68:].strip()
        })
    elif type_field == "F" and commented != "*":
        # Define tokens for F-spec
        filename = code_line[start_pos:start_pos+8].strip()
        if filename:  # Filename is present
            tokens.update({
                "filename": filename,
                "file_type": code_line[start_pos+8:start_pos+9].strip(),
                "designation": code_line[start_pos+9:start_pos+10].strip(),
                "eof": code_line[start_pos+10:start_pos+11].strip(),
                "sequence": code_line[start_pos+11:start_pos+12].strip(),
                "format": code_line[start_pos+13:start_pos+14].strip(),
                "record_length": code_line[start_pos+17:start_pos+21].strip(),
                "limit": code_line[start_pos+21:start_pos+22].strip(),
                "key_lenght": code_line[start_pos+22:start_pos+23].strip(),
                "adr_type": code_line[start_pos+23:start_pos+24].strip(),
                "organization": code_line[start_pos+24:start_pos+25].strip(),
                "overflow_indicator": code_line[start_pos+26:start_pos+28].strip(),
                "key_location": code_line[start_pos+28:start_pos+32].strip(),
                "extension": code_line[start_pos+32:start_pos+33].strip(),
                "device": code_line[start_pos+33:start_pos+40].strip(),
                "continuation": code_line[start_pos+46:start_pos+47].strip(),
                "routine": code_line[start_pos+47:start_pos+53].strip(),
                "entry": code_line[start_pos+53:start_pos+59].strip(),
                "add": code_line[start_pos+59:start_pos+60].strip(),
                "condition": code_line[start_pos+64:start_pos+66].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
        else:  # Filename is blank, handle extended record format
            tokens.update({
                "ext_record": code_line[start_pos+12:start_pos+22].strip(),
                "recoard_number": code_line[start_pos+40:start_pos+46].strip(),
                "key": code_line[start_pos+46:start_pos+47].strip(),
                "option": code_line[start_pos+47:start_pos+53].strip(),
                "entry": code_line[start_pos+51:start_pos+61].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
    elif type_field == "E" and commented != "*":
        # Define tokens for E-spec
        tokens.update({
            "from_file": code_line[start_pos+4:start_pos+12].strip(),
            "to_file": code_line[start_pos+12:start_pos+20].strip(),
            "table_name": code_line[start_pos+20:start_pos+26].strip(),
            "number_records": code_line[start_pos+26:start_pos+29].strip(),
            "table_entry": code_line[start_pos+29:start_pos+33].strip(),
            "length_of_data": code_line[start_pos+33:start_pos+36].strip(),
            "format_of_data": code_line[start_pos+36:start_pos+37].strip(),
            "data_decimal": code_line[start_pos+37:start_pos+38].strip(),
            "data_sequence": code_line[start_pos+38:start_pos+39].strip(),            
            "other_name": code_line[start_pos+39:start_pos+52].strip(),
            "length": code_line[start_pos+52:start_pos+55].strip(),
            "format": code_line[start_pos+55:start_pos+56].strip(),
            "decimal": code_line[start_pos+56:start_pos+57].strip(),
            "sequence": code_line[start_pos+57:start_pos+58].strip(),
            "comments": code_line[start_pos+51:].strip()
        })
    elif type_field == "L" and commented != "*":
        # Define tokens for L-spec
        tokens.update({
            "filename": code_line[start_pos:start_pos+8].strip(),
            "line_number": code_line[start_pos+8:start_pos+11].strip(),
            "paper_lenght": code_line[start_pos+11:start_pos+13].strip(),
            "line_number_overflow": code_line[start_pos+13:start_pos+16].strip(),
            "line_overflow": code_line[start_pos+16:start_pos+18].strip(),
            "comments": code_line[start_pos+68:].strip()
        }) 
    elif type_field == "I" and commented != "*":
        # Define tokens for I-spec
        ds = code_line[start_pos+12:start_pos+14].strip()
        field = code_line[start_pos+1:start_pos+14].strip()
        format = code_line[start_pos+36:start_pos+37].strip()
        if ds == "DS":  # Data structure exist
            tokens.update({
                "ds_name": code_line[start_pos:start_pos+6],  
                "external": code_line[start_pos+10:start_pos+11].strip(),
                "option": code_line[start_pos+10:start_pos+11].strip(),
                "ds": ds,
                "ext_file": code_line[start_pos+15:start_pos+25].strip(),
                "occur": code_line[start_pos+25:start_pos+29].strip(),
                "length": code_line[start_pos+29:start_pos+34].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
        elif format == "C":  # Named constant 
            tokens.update({
                "constant": code_line[start_pos+14:start_pos+28].strip(),
                "constant_value": code_line[start_pos+36:start_pos+80].strip(),
                "constant_name": code_line[start_pos+46:start_pos+52].strip(), 
                "comments": code_line[start_pos+68:].strip()           
            })  
        elif field == "" and format != "C":  # other fields type
            tokens.update({
                "init": code_line[start_pos+1:start_pos+2].strip(),
                "ext_field_name": code_line[start_pos+14:start_pos+24],
                "data_format": format,
                "from_position": code_line[start_pos+37:start_pos+41].strip(),
                "to_position": code_line[start_pos+41:start_pos+45].strip(),
                "decimal_precision": code_line[start_pos+45:start_pos+46].strip(),
                "field_name": code_line[start_pos+46:start_pos+52].strip(),
                "control_level": code_line[start_pos+52:start_pos+54].strip(),
                "matching_field": code_line[start_pos+54:start_pos+56].strip(),
                "relation_field": code_line[start_pos+56:start_pos+58].strip(),
                "positive_field": code_line[start_pos+58:start_pos+60].strip(),
                "negative_field": code_line[start_pos+60:start_pos+62].strip(),
                "zero_blank": code_line[start_pos+62:start_pos+64].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
        else:
            tokens.update({
                "filename": code_line[start_pos:start_pos+8].strip(),  
                "sequence": code_line[start_pos+8:start_pos+10].strip(),
                "number": code_line[start_pos+10:start_pos+11].strip(),  
                "record_id": code_line[start_pos+12:start_pos+14].strip(),
                "first_position": code_line[start_pos+14:start_pos+18].strip(),
                "first_non": code_line[start_pos+18:start_pos+19].strip(),
                "first_code_part": code_line[start_pos+19:start_pos+20].strip(),
                "first_char": code_line[start_pos+20:start_pos+21].strip(),
                "second_position": code_line[start_pos+21:start_pos+25].strip(),
                "second_non": code_line[start_pos+25:start_pos+26].strip(),
                "second_code_part": code_line[start_pos+26:start_pos+27].strip(),
                "second_char": code_line[start_pos+27:start_pos+28].strip(),
                "third_position": code_line[start_pos+28:start_pos+32].strip(),
                "third_non": code_line[start_pos+32:start_pos+33].strip(),
                "third_code_part": code_line[start_pos+33:start_pos+34].strip(),
                "third_char": code_line[start_pos+34:start_pos+35].strip(),
                "comments": code_line[start_pos+68:].strip()
            })                
    elif type_field == "C" and commented != "*":
        # Define tokens for C-spec
        tokens.update({
            "control": code_line[start_pos:start_pos+2].strip(),
            "indicator1": code_line[start_pos+2:start_pos+5].strip(),
            "indicator2": code_line[start_pos+5:start_pos+8].strip(),
            "indicator3": code_line[start_pos+8:start_pos+11].strip(),
            "factor1": code_line[start_pos+11:start_pos+21].strip(),
            "opcode": code_line[start_pos+21:start_pos+26].strip(),
            "factor2": code_line[start_pos+26:start_pos+36].strip(),
            "result": code_line[start_pos+36:start_pos+42].strip(),
            "len": code_line[start_pos+42:start_pos+45].strip(),
            "de": code_line[start_pos+45:start_pos+47].strip(),
            "hi": code_line[start_pos+47:start_pos+49].strip(),
            "lo": code_line[start_pos+49:start_pos+51].strip(),
            "eq": code_line[start_pos+51:start_pos+53].strip(),
            "comments": code_line[start_pos+53:].strip()
        })
    elif type_field == "O" and commented != "*":
        add_del = code_line[start_pos+9:start_pos+12].strip()
        and_or = code_line[start_pos+7:start_pos+10].strip()
        named = code_line[start_pos+1:start_pos+16].strip()
        # Define tokens for O-spec
        # Disk output
        if add_del == "ADD" or add_del == "DEL":
            tokens.update({
                "name": code_line[start_pos:start_pos+8].strip(),
                "type": code_line[start_pos+8:start_pos+9].strip(),
                "add_del": add_del,
                "indicator1": code_line[start_pos+16:start_pos+19].strip(),
                "indicator2": code_line[start_pos+19:start_pos+22].strip(),
                "indicator3": code_line[start_pos+22:start_pos+25].strip(),
                "excpt_name": code_line[start_pos+25:start_pos+31].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
        # Additional record 
        elif and_or == "AND" or and_or == "OR":
            tokens.update({
                "and_or": and_or,
                "before_space": code_line[start_pos+10:start_pos+11].strip(),
                "after_space": code_line[start_pos+11:start_pos+12].strip(),
                "before_skip": code_line[start_pos+12:start_pos+14].strip(),
                "after_skip": code_line[start_pos+14:start_pos+16].strip(),
                "indicator1": code_line[start_pos+16:start_pos+19].strip(),
                "indicator2": code_line[start_pos+19:start_pos+22].strip(),
                "indicator3": code_line[start_pos+22:start_pos+25].strip(),
                "excpt_name": code_line[start_pos+25:start_pos+31].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
        # Field output    
        elif not named:
            tokens.update({
                "indicator1": code_line[start_pos+16:start_pos+19].strip(),
                "indicator2": code_line[start_pos+19:start_pos+22].strip(),
                "indicator3": code_line[start_pos+22:start_pos+25].strip(),
                "field_name": code_line[start_pos+25:start_pos+31].strip(),
                "editcode": code_line[start_pos+31:start_pos+32].strip(),
                "after_blank": code_line[start_pos+32:start_pos+33].strip(),
                "end_position": code_line[start_pos+33:start_pos+37].strip(),
                "type": code_line[start_pos+37:start_pos+38].strip(),
                "editword_constant": code_line[start_pos+38:start_pos+64].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
        # Record output     
        else:
            tokens.update({
                "name": code_line[start_pos:start_pos+8].strip(),
                "type": code_line[start_pos+8:start_pos+9].strip(),
                "anticipated_call": code_line[start_pos+9:start_pos+10].strip(),
                "before_space": code_line[start_pos+10:start_pos+11].strip(),
                "after_space": code_line[start_pos+11:start_pos+12].strip(),
                "before_skip": code_line[start_pos+12:start_pos+14].strip(),
                "after_skip": code_line[start_pos+14:start_pos+16].strip(),
                "indicator1": code_line[start_pos+16:start_pos+19].strip(),
                "indicator2": code_line[start_pos+19:start_pos+22].strip(),
                "indicator3": code_line[start_pos+22:start_pos+25].strip(),
                "excpt_name": code_line[start_pos+25:start_pos+31].strip(),
                "comments": code_line[start_pos+68:].strip()
            })
    else:        
        tokens.update({
            "comments": code_line[start_pos+2:].strip()
            })                                                               
    pass tokens
