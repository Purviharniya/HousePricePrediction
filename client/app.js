function getValues(name) {
  var values = document.getElementsByName(name);
  for (var i in values) {
    if (values[i].checked) {
      return parseInt(i) + 1;
    }
  }
  return -1;
}

// function getBATH(){
//     var baths = document.getElementsByName('bath');
//     for(var i in baths){
//         if(baths[i].checked){
//             return parseInt(i)+1;
//         }
//     }
//     return -1;
// }

function EstimatePrice() {
  console.log("Estimating the price...");
  var sqft = document.getElementById("area_sqft");
  var bhk = getValues("bhk");
  var bath = getValues("bath");
  var location = document.getElementById("SLocations");
  var estPrice = document.getElementById("EstimatedPrice");

  var url = "http://127.0.0.1:5000/predict";

  $.post(
    url,
    {
      total_sqft: parseFloat(sqft.value),
      location: location.value,
      bath: bath,
      bhk: bhk,
    },
    function (data, status) {
      console.log("estimated price:", data.estimated_price);
      estPrice.innerHTML =
        "<h3 class='predictor'>Predicted Price: Rs. " + data.estimated_price.toString() + " Lakh </h3>";
      console.log("Status:", status);
    }
  );
}

function onPageLoad() {
  console.log("hi.. document loaded");
  var url = "http://127.0.0.1:5000/get_loc_names";
  $.get(url, function (data, status) {
    console.log("getting the locations from the server..");
    if (data) {
      var locations = data.locations;
      var sloc = document.getElementById("SLocations");
      $("#SLocations").empty();
      for (var i in locations) {
        var opt = new Option(locations[i]);
        // console.log(opt);
        $("#SLocations").append(opt);
      }
    }
  });
}

window.onload = onPageLoad;
