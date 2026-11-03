# Common Markdown Syntax Reference

## Headings

Use number signs (#) to indicate headings. Number sign (#) can be used to indicate level-1 to level-6 headings. A level-1 heading corresponds to one number sign (#), a level-2 heading corresponds to two number signs (#), and so on.

```
# Level-1 heading
## Level-2 heading
### Level-3 heading
#### Level-4 heading
##### Level-5 heading
###### Level-6 heading
```

## Fonts

- Italic: Use one asterisk (*) to indicate italic.

  ```
  *Italic text*
  ```

- Bold: Use two asterisks (**) to indicate bold.

  ```
  **Bold text**
  ```

- Bold italic: Use three asterisks (***) to indicate bold italic.

  ```
  ***Bold italic text***
  ```

## Lists

- Unordered list: Use asterisks (*), plus signs (+), or hyphens (-) as list markers. A space must be added after the marker, and then the content can be entered. It is recommended that you use the same symbol in an unordered list.

  ```
  * Item 1
  * Item 2
  * Item 3
   
  + Item 1
  + Item 2
  + Item 3
   
   
  - Item 1
  - Item 2
  - Item 3
  ```

- Ordered list: Use numbers followed by a period (.).  

  ```
  1. Item 1
  2. Item 2
  3. Item 3
  ```

- Nested list: Add four spaces before the options in the sublist.

  ```
  1. Item 1:
      - First element nested in item 1
      - Second element nested in item 1
  2. Item 2:
      - First element nested in item 2
      - Second element nested in item 2
  ```

## References

Markdown block reference: Use a greater-than sign (>), followed by a space.

```
> NOTE
```

## Code

- Inline code: For a function or code snippet in a paragraph, use a pair of backticks (``) to enclose the function or code snippet.

```
`printf()` function
```

- Code block: Use three backticks (```) to enclose the code.

```
select * from table;
```

## Links

The link format is as follows:

```
[Link name](Link address)
```

## Images

The image usage is as follows:

```
![alt Attribute text](Image address)
 
![alt Attribute text](Image address "Optional title")
```

- Start with an exclamation mark (!).
- Then, use a pair of brackets [], which contains the alternative text of the image.
- End with a pair of parentheses (), which contains the URL of the image. You can also use a pair of quotation marks (") to enclose the optional title attribute.

## Tables

In Markdown, you can use vertical bars (|) to separate different cells and use hyphens (-) to separate the table header row from other rows. The syntax format is as follows:

```
|  Table header   | Table header  |
|  ----  | ----  |
| Cell  | Cell |
| Cell  | Cell |
```

You can set the alignment mode of tables as follows:

- -: Align the content and title bar to the right.
- :- Align the content and title bar to the left.
- :-: Align the content and title bar to the center.
