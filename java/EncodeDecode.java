import java.util.Scanner; // For user input.

public class EncodeDecode {
    static boolean valid_character (char c) {
	// Characters from ' ' to '~' are considered valid_characters
	if (c <= 126 && c >= 32) {
	    return true;
	} else {
	    return false;
	}
    }
    
    static String encode (String input, int offset) {
	String output = "";
	char temp_char;
	for (int i = 0; i < input.length(); i++) {
	    temp_char = (char) (input.charAt(i) + offset);
	    // If the output would not be a valid character, then subtract 95.
	    if (!valid_character(temp_char)) {
		output += (char) (temp_char - 95);
	    } else {
		output += temp_char;
	    }
	}
	
	return output;
    }

    static String decode (String input, int offset) {
	String output = "";
	char temp_char;
	
	for (int i = 0; i < input.length(); i++) {
	    temp_char = (char) (input.charAt(i) - offset);
	    // If the output would not be a valid character, then add 95.
	    if (!valid_character(temp_char)) {
		output += (char) (temp_char + 95);
	    } else {
		output += temp_char;
	    }
	}
	
	return output;
    }
    
    public static void main (String[] args) {
	Scanner input = new Scanner(System.in);
	while (true) {
	    // d for decode, encode is default.
	    System.out.println("Encode or Decode? [E/d]");

	    String action = input.nextLine();
	    
	    System.out.print("Enter the key: ");
	    int key = input.nextInt();
	    // This consumes the newline from after the nextInt();
	    input.nextLine();
	    
	    if (action.equals("d") || action.equals("decode") || action.equals("D") || action.equals("Decode")) {
		System.out.print("Enter the string that you would like to decode: ");
		System.out.println(decode(input.nextLine(), key));
	    } else {		
		System.out.print("Enter the string that you would like to encode: ");
		System.out.println(encode(input.nextLine(), key));
	    }
	}
    }
}
