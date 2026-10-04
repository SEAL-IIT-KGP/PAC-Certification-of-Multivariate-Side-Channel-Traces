-- Older graphicx has no alt= key: drop the image alt text that pandoc 3.8 emits.
function Image(im)
  im.caption = {}
  return im
end
