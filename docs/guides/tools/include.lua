-- A code block with attribute `transcript=PATH` is replaced by the file's contents.
-- Missing files become a visible placeholder so the PDF still builds.
function CodeBlock(cb)
  local path = cb.attributes["transcript"]
  if not path then return cb end
  local f = io.open(path, "r")
  if not f then
    return pandoc.Para({pandoc.Strong(pandoc.Str("[transcript not recorded yet: " .. path .. "]"))})
  end
  local text = f:read("a"); f:close()
  text = text:gsub("[%z\1-\8\11\12\14-\31\127]", "")   -- drop control characters
  -- hard-wrap long lines so they stay inside the page
  local width = 108
  local out = {}
  for line in (text .. "\n"):gmatch("(.-)\n") do
    while #line > width do
      table.insert(out, line:sub(1, width))
      line = "    " .. line:sub(width + 1)
    end
    table.insert(out, line)
  end
  return pandoc.CodeBlock(table.concat(out, "\n"), pandoc.Attr("", {"text"}, {}))
end
-- Images whose file is missing become a placeholder.
function Image(im)
  local f = io.open(im.src, "r")
  if f then f:close() return im end
  return pandoc.Strong(pandoc.Str("[figure not produced yet: " .. im.src .. "]"))
end
