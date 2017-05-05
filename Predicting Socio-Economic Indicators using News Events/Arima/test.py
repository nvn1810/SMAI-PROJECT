import mechanize
br = mechanize.Browser()
url = "http://agmarknet.nic.in/agnew/NationalBEnglish/SpecificCommodityWeeklyReport.aspx"
response = br.open(url)
#print response.read()      # the text of the page
response1 = br.response()  # get the response again
#print response1.read()     # can apply lxml.html.fromstring()
# for form in br.forms():
# 	print "Form name:", form.name
# 	print form
br.select_form("form1")
for control in br.form.controls:
    print control
    print "type=%s, name=%s value=%s" % (control.type, control.name, br[control.name])

control = br.form.find_control(control.name)
if control.type == "select":  # means it's class ClientForm.SelectControl
    for item in control.items:
    	print " name=%s values=%s" % (item.name, str([label.text  for label in item.get_labels()]))