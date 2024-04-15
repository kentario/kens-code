import java.util.*;

public class Variadic {
    static void printf (String fmt, String... args) throws Exception {
	int arg = 0;
	char[] chars = fmt.toCharArray();
	for (int i = 0; i < chars.length; i++) {
	    if (chars[i] == '%') {
		if (++i < chars.length) {
		    switch (chars[i]) {
		    case 's':
			// More formats than arguments.
			if (arg >= args.length) {
			    throw new Exception("Erorr, no matching argument to format at character " + i);
			}
			System.out.print(args[arg]);
			arg++;
			break;
		    case '%':
			System.out.print("%");
		    }
		}
	    } else {
		System.out.print(chars[i]);
	    }
	}
    }
    
    public static void main (String[] args) {
	try {
	    printf("hi, format: %s, another one: %s\n%s\n", "first", "second", "third");
	} catch (Exception e) {
	    e.printStackTrace();
	}
    }
}
